import numpy as np
import torch
import time
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from baseline.relation_cgeom.encoder_three_branch_hyp_relation_cgeom import TSEncoderThreeBranchHypRelationCGeom
from baseline.relation_cgeom.losses_complex_geom import complex_avg_pool1d
from baseline.relation_cgeom.losses_eeg_multibranch import EEGMultiBranchLoss, EEGMultiBranchLossConfig
from baseline.relation_cgeom.utils import centerize_vary_length_series, split_with_nan, take_per_row, torch_pad_nan


class TS2VecThreeBranchHypRelationCGeom:
    '''TS2Vec with hyp-relation structure aligned to the complex_geom loss.'''

    def __init__(
        self,
        input_dims,
        output_dims=320,
        hidden_dims=64,
        depth=10,
        device='cuda',
        lr=0.001,
        batch_size=16,
        max_train_length=None,
        temporal_unit=0,
        distributed=False,
        sampling_rate=256.0,
        bands=((8.0, 13.0), (13.0, 20.0), (20.0, 30.0)),
        use_raw_branch=True,
        use_complex_branch=True,
        use_hyperbolic_branch=True,
        share_band_encoder=False,
        band_fusion_type='concat_linear',
        tf_fusion_type='concat_linear',
        global_fusion_type='concat_linear',
        hyperbolic_depth=1,
        hyperbolic_curvature=1.0,
        learnable_curvature=True,
        tf_align_weight=0.3,
        tf_align_type='cosine',
        raw_mask_weight=0.0,
        raw_mask_type='cosine',
        after_iter_callback=None,
        after_epoch_callback=None,
        show_progress=False,
    ):
        super().__init__()
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.max_train_length = max_train_length
        self.temporal_unit = temporal_unit
        self.distributed = distributed

        self._model = TSEncoderThreeBranchHypRelationCGeom(
            input_dims=input_dims,
            output_dims=output_dims,
            hidden_dims=hidden_dims,
            depth=depth,
            sampling_rate=sampling_rate,
            bands=bands,
            use_raw_branch=use_raw_branch,
            use_complex_branch=use_complex_branch,
            use_hyperbolic_branch=use_hyperbolic_branch,
            share_band_encoder=share_band_encoder,
            band_fusion_type=band_fusion_type,
            tf_fusion_type=tf_fusion_type,
            global_fusion_type=global_fusion_type,
            hyperbolic_depth=hyperbolic_depth,
            hyperbolic_curvature=hyperbolic_curvature,
            learnable_curvature=learnable_curvature,
            use_raw_mask_loss=raw_mask_weight > 0,
        ).to(self.device)
        if self.distributed:
            self._train_net = torch.nn.parallel.DistributedDataParallel(
                self._model,
                device_ids=[self.device.index],
                output_device=self.device.index,
            )
        else:
            self._train_net = self._model
        self.net = torch.optim.swa_utils.AveragedModel(self._model)
        self.net.update_parameters(self._model)

        self.loss_module = EEGMultiBranchLoss(
            EEGMultiBranchLossConfig(
                lambda_cgeom=1.0,
                lambda_tf_align=tf_align_weight,
                lambda_raw_mask=raw_mask_weight,
                use_cgeom_main=True,
                use_tf_align=use_raw_branch,
                use_raw_mask=use_raw_branch and raw_mask_weight > 0,
                tf_align_type=tf_align_type,
                raw_mask_type=raw_mask_type,
            )
        )

        self.after_iter_callback = after_iter_callback
        self.after_epoch_callback = after_epoch_callback
        self.show_progress = show_progress

        self.n_epochs = 0
        self.n_iters = 0
        self.train_history = []

    def _compute_pair_loss(self, x, net=None):
        if net is None:
            net = self._train_net

        ts_l = x.size(1)
        crop_l = np.random.randint(low=2 ** (self.temporal_unit + 1), high=ts_l + 1)
        crop_left = np.random.randint(ts_l - crop_l + 1)
        crop_right = crop_left + crop_l
        crop_eleft = np.random.randint(crop_left + 1)
        crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
        crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=x.size(0))

        view1 = net(
            take_per_row(x, crop_offset + crop_eleft, crop_right - crop_eleft),
            return_aux=True,
        )
        view1['main_repr'] = view1['main_repr'][:, -crop_l:]

        view2 = net(
            take_per_row(x, crop_offset + crop_left, crop_eright - crop_left),
            return_aux=True,
        )
        view2['main_repr'] = view2['main_repr'][:, :crop_l]

        return self.loss_module(view1, view2, temporal_unit=self.temporal_unit)

    def evaluate_loss(self, eval_data, batch_size=None, verbose=False, desc='Pretrain eval'):
        assert eval_data.ndim == 3
        if batch_size is None:
            batch_size = self.batch_size

        eval_data = eval_data[~np.isnan(eval_data).all(axis=2).all(axis=1)]
        eval_dataset = TensorDataset(torch.from_numpy(eval_data).to(torch.float))
        eval_loader = DataLoader(eval_dataset, batch_size=min(batch_size, len(eval_dataset)), shuffle=False, drop_last=False)

        org_training = self.net.training
        self.net.eval()

        loss_sum = 0.0
        n_iters = 0
        term_sums = {}
        iterator = eval_loader
        if verbose or self.show_progress:
            iterator = tqdm(eval_loader, desc=desc, leave=True, dynamic_ncols=True)

        with torch.no_grad():
            for batch in iterator:
                x = batch[0]
                if self.max_train_length is not None and x.size(1) > self.max_train_length:
                    window_offset = (x.size(1) - self.max_train_length) // 2
                    x = x[:, window_offset: window_offset + self.max_train_length]
                x = x.to(self.device)

                loss, loss_terms = self._compute_pair_loss(x, net=self.net)
                loss_sum += float(loss.item())
                n_iters += 1
                for key, value in loss_terms.items():
                    term_sums[key] = term_sums.get(key, 0.0) + float(value)

                if verbose or self.show_progress:
                    postfix = {'loss': f'{loss.item():.6f}'}
                    if 'loss_cgeom' in loss_terms:
                        postfix['cgeom'] = f"{loss_terms['loss_cgeom']:.6f}"
                    if 'loss_tf_align' in loss_terms:
                        postfix['tf_align'] = f"{loss_terms['loss_tf_align']:.6f}"
                    iterator.set_postfix(postfix)

        self.net.train(org_training)
        if n_iters < 1:
            raise RuntimeError('No eval batches were processed.')

        record = {'eval_loss_total': float(loss_sum / n_iters)}
        for key, value in term_sums.items():
            record[f'eval_{key}'] = float(value / n_iters)
        return record

    def fit(self, train_data, n_epochs=None, n_iters=None, verbose=False):
        assert train_data.ndim == 3

        if n_iters is None and n_epochs is None:
            n_iters = 200 if train_data.size <= 100000 else 600

        if self.max_train_length is not None:
            sections = train_data.shape[1] // self.max_train_length
            if sections >= 2:
                train_data = np.concatenate(split_with_nan(train_data, sections, axis=1), axis=0)

        temporal_missing = np.isnan(train_data).all(axis=-1).any(axis=0)
        if temporal_missing[0] or temporal_missing[-1]:
            train_data = centerize_vary_length_series(train_data)

        train_data = train_data[~np.isnan(train_data).all(axis=2).all(axis=1)]
        train_dataset = TensorDataset(torch.from_numpy(train_data).to(torch.float))
        train_sampler = None
        if self.distributed:
            train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=min(self.batch_size, len(train_dataset)),
            shuffle=train_sampler is None,
            sampler=train_sampler,
            drop_last=True,
        )

        optimizer = torch.optim.AdamW(self._train_net.parameters(), lr=self.lr)
        loss_log = []

        while True:
            if n_epochs is not None and self.n_epochs >= n_epochs:
                break
            if train_sampler is not None:
                train_sampler.set_epoch(self.n_epochs)

            cum_loss = 0
            n_epoch_iters = 0
            interrupted = False
            epoch_term_sums = {}
            epoch_start_time = time.time()

            iterator = train_loader
            if self.show_progress:
                iterator = tqdm(
                    train_loader,
                    desc=f"Pretrain epoch {self.n_epochs}",
                    leave=True,
                    dynamic_ncols=True,
                )

            for batch in iterator:
                if n_iters is not None and self.n_iters >= n_iters:
                    interrupted = True
                    break

                x = batch[0]
                if self.max_train_length is not None and x.size(1) > self.max_train_length:
                    window_offset = np.random.randint(x.size(1) - self.max_train_length + 1)
                    x = x[:, window_offset: window_offset + self.max_train_length]
                x = x.to(self.device)

                optimizer.zero_grad()
                loss, loss_terms = self._compute_pair_loss(x)

                loss.backward()
                optimizer.step()
                self.net.update_parameters(self._model)

                cum_loss += loss.item()
                n_epoch_iters += 1
                self.n_iters += 1
                for key, value in loss_terms.items():
                    epoch_term_sums[key] = epoch_term_sums.get(key, 0.0) + float(value)

                if self.show_progress:
                    postfix = {'loss': f'{loss.item():.6f}'}
                    if 'loss_cgeom' in loss_terms:
                        postfix['cgeom'] = f"{loss_terms['loss_cgeom']:.6f}"
                    if 'loss_tf_align' in loss_terms:
                        postfix['tf_align'] = f"{loss_terms['loss_tf_align']:.6f}"
                    iterator.set_postfix(postfix)

                if self.after_iter_callback is not None:
                    self.after_iter_callback(self, loss.item())

            if interrupted:
                break

            cum_loss /= n_epoch_iters
            loss_log.append(cum_loss)
            epoch_record = {
                'epoch': self.n_epochs,
                'loss_total': float(cum_loss),
                'epoch_time_sec': float(time.time() - epoch_start_time),
            }
            for key, value in epoch_term_sums.items():
                epoch_record[key] = float(value / n_epoch_iters)
            self.train_history.append(epoch_record)
            if verbose:
                loss_msg = f"Epoch #{self.n_epochs}: loss={cum_loss} time={epoch_record['epoch_time_sec']:.2f}s"
                if 'loss_tf_align' in epoch_record:
                    loss_msg += f" tf_align={epoch_record['loss_tf_align']:.6f}"
                if 'loss_raw_mask' in epoch_record:
                    loss_msg += f" raw_mask={epoch_record['loss_raw_mask']:.6f}"
                if 'loss_cgeom' in epoch_record:
                    loss_msg += f" cgeom={epoch_record['loss_cgeom']:.6f}"
                print(loss_msg)
            self.n_epochs += 1

            if self.after_epoch_callback is not None:
                self.after_epoch_callback(self, epoch_record)

        return loss_log

    def _eval_with_pooling(self, x, mask=None, slicing=None, encoding_window=None):
        out = self.net(x.to(self.device, non_blocking=True), mask)
        if encoding_window == 'full_series':
            if slicing is not None:
                out = out[:, slicing]
            out = complex_avg_pool1d(out, kernel_size=out.size(1))
        elif isinstance(encoding_window, int):
            out = complex_avg_pool1d(
                out,
                kernel_size=encoding_window,
                stride=1,
                padding=encoding_window // 2,
            )
            if encoding_window % 2 == 0:
                out = out[:, :-1]
            if slicing is not None:
                out = out[:, slicing]
        elif encoding_window == 'multiscale':
            p = 0
            reprs = []
            while (1 << p) + 1 < out.size(1):
                t_out = complex_avg_pool1d(
                    out,
                    kernel_size=(1 << (p + 1)) + 1,
                    stride=1,
                    padding=1 << p,
                )
                if slicing is not None:
                    t_out = t_out[:, slicing]
                reprs.append(t_out)
                p += 1
            out = torch.cat(reprs, dim=-1)
        else:
            if slicing is not None:
                out = out[:, slicing]

        return out.cpu()

    def encode(self, data, mask=None, encoding_window=None, causal=False, sliding_length=None, sliding_padding=0, batch_size=None):
        assert self.net is not None, 'please train or load a net first'
        assert data.ndim == 3
        if batch_size is None:
            batch_size = self.batch_size
        n_samples, ts_l, _ = data.shape

        org_training = self.net.training
        self.net.eval()

        dataset = TensorDataset(torch.from_numpy(data).to(torch.float))
        loader = DataLoader(dataset, batch_size=batch_size)

        with torch.no_grad():
            output = []
            for batch in tqdm(loader, desc='Encode', leave=True):
                x = batch[0]
                if sliding_length is not None:
                    reprs = []
                    if n_samples < batch_size:
                        calc_buffer = []
                        calc_buffer_l = 0
                    for i in range(0, ts_l, sliding_length):
                        l = i - sliding_padding
                        r = i + sliding_length + (sliding_padding if not causal else 0)
                        x_sliding = torch_pad_nan(
                            x[:, max(l, 0): min(r, ts_l)],
                            left=-l if l < 0 else 0,
                            right=r - ts_l if r > ts_l else 0,
                            dim=1,
                        )
                        if n_samples < batch_size:
                            if calc_buffer_l + n_samples > batch_size:
                                out = self._eval_with_pooling(
                                    torch.cat(calc_buffer, dim=0),
                                    mask,
                                    slicing=slice(sliding_padding, sliding_padding + sliding_length),
                                    encoding_window=encoding_window,
                                )
                                reprs += torch.split(out, n_samples)
                                calc_buffer = []
                                calc_buffer_l = 0
                            calc_buffer.append(x_sliding)
                            calc_buffer_l += n_samples
                        else:
                            out = self._eval_with_pooling(
                                x_sliding,
                                mask,
                                slicing=slice(sliding_padding, sliding_padding + sliding_length),
                                encoding_window=encoding_window,
                            )
                            reprs.append(out)

                    if n_samples < batch_size and calc_buffer_l > 0:
                        out = self._eval_with_pooling(
                            torch.cat(calc_buffer, dim=0),
                            mask,
                            slicing=slice(sliding_padding, sliding_padding + sliding_length),
                            encoding_window=encoding_window,
                        )
                        reprs += torch.split(out, n_samples)

                    out = torch.cat(reprs, dim=1)
                    if encoding_window == 'full_series':
                        out = complex_avg_pool1d(out, kernel_size=out.size(1)).squeeze(1)
                else:
                    out = self._eval_with_pooling(x, mask, encoding_window=encoding_window)
                    if encoding_window == 'full_series':
                        out = out.squeeze(1)

                output.append(out)

            output = torch.cat(output, dim=0)

        self.net.train(org_training)
        return output.numpy()

    def save(self, fn):
        torch.save(self.net.state_dict(), fn)

    def load(self, fn):
        state_dict = torch.load(fn, map_location=self.device)
        self.net.load_state_dict(state_dict)
