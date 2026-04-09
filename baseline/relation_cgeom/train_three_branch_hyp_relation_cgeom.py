import argparse
import datetime
import os
import time

import datautils
import tasks

from ts2vec_three_branch_hyp_relation_cgeom import TS2VecThreeBranchHypRelationCGeom
from utils import (
    broadcast_object,
    cleanup_distributed,
    init_distributed_training,
    name_with_datetime,
    pkl_save,
    sync_barrier,
)


def parse_band_spec(spec: str):
    bands = []
    for chunk in spec.split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        low, high = chunk.split('-', 1)
        bands.append((float(low), float(high)))
    if not bands:
        raise ValueError(f'Invalid band specification: {spec}')
    return tuple(bands)


def save_checkpoint_callback(save_every=1, unit='epoch'):
    assert unit in ('epoch', 'iter')

    def callback(model, loss):
        n = model.n_epochs if unit == 'epoch' else model.n_iters
        if n % save_every == 0:
            model.save(f'{run_dir}/model_{n}.pkl')

    return callback


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='The EEG dataset name')
    parser.add_argument('run_name', help='The folder name used to save model and evaluation metrics.')
    parser.add_argument('--loader', type=str, required=True, help='Currently supports eeg_cls')
    parser.add_argument('--gpu', type=int, nargs='+', default=[0], help='One or more GPU ids used for training. Use torchrun for multi-GPU.')
    parser.add_argument('--batch-size', type=int, default=8, help='The batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate')
    parser.add_argument('--repr-dims', type=int, default=320, help='The representation dimension')
    parser.add_argument('--hidden-dims', type=int, default=64, help='The hidden dimension')
    parser.add_argument('--depth', type=int, default=10, help='The number of residual blocks per branch')
    parser.add_argument('--max-train-length', type=int, default=3000, help='Max cropped sequence length')
    parser.add_argument('--iters', type=int, default=None, help='The number of iterations')
    parser.add_argument('--epochs', type=int, default=None, help='The number of epochs')
    parser.add_argument('--save-every', type=int, default=None, help='Save the checkpoint every <save_every> iterations/epochs')
    parser.add_argument('--seed', type=int, default=None, help='The random seed')
    parser.add_argument('--max-threads', type=int, default=None, help='The maximum allowed number of threads used by this process')
    parser.add_argument('--eval', action='store_true', help='Whether to perform evaluation after training')
    parser.add_argument('--sampling-rate', type=float, default=256.0, help='Effective EEG sampling rate after preprocessing')
    parser.add_argument('--bands', type=str, default='8-13,13-20,20-30', help='Comma-separated band list, e.g. 8-13,13-20,20-30')
    parser.add_argument('--share-band-encoder', action='store_true', help='Share one complex encoder across all filter-bank bands')
    parser.add_argument('--disable-raw-branch', action='store_true', help='Disable the raw temporal branch')
    parser.add_argument('--disable-hyperbolic-branch', action='store_true', help='Disable the hyperbolic relation branch')
    parser.add_argument('--band-fusion-type', type=str, default='concat_linear', choices=['concat_linear', 'gated_sum'])
    parser.add_argument('--hyperbolic-depth', type=int, default=1, help='Depth of the hyperbolic branch')
    parser.add_argument('--hyperbolic-curvature', type=float, default=1.0, help='Initial curvature for the hyperbolic branch')
    parser.add_argument('--fixed-curvature', action='store_true', help='Use a fixed hyperbolic curvature instead of learning it')
    parser.add_argument('--tf-align-weight', type=float, default=0.3, help='Weight of the raw-complex alignment loss')
    parser.add_argument('--tf-align-type', type=str, default='cosine', choices=['cosine', 'mse'], help='Alignment loss type')
    parser.add_argument('--raw-mask-weight', type=float, default=0.0, help='Weight of the raw masked-representation consistency loss')
    parser.add_argument('--raw-mask-type', type=str, default='cosine', choices=['cosine', 'mse'], help='Raw masked-representation loss type')
    args = parser.parse_args()

    if args.loader != 'eeg_cls':
        raise ValueError(f'Unknown loader {args.loader}. train_three_branch_hyp_relation_cgeom.py currently supports only eeg_cls.')

    env = init_distributed_training(args.gpu, seed=args.seed, max_threads=args.max_threads)
    device = env['device']
    distributed = env['distributed']
    is_main = env['is_main_process']

    if is_main:
        print('Dataset:', args.dataset)
        print('Arguments:', str(args))
        print('Loading data... ', end='')

    train_data, train_labels, test_data, test_labels = datautils.load_EEG_cls(args.dataset)

    if is_main:
        print('done')

    config = dict(
        batch_size=args.batch_size,
        lr=args.lr,
        output_dims=args.repr_dims,
        hidden_dims=args.hidden_dims,
        depth=args.depth,
        max_train_length=args.max_train_length,
        sampling_rate=args.sampling_rate,
        bands=parse_band_spec(args.bands),
        use_raw_branch=not args.disable_raw_branch,
        use_complex_branch=True,
        use_hyperbolic_branch=not args.disable_hyperbolic_branch,
        share_band_encoder=args.share_band_encoder,
        band_fusion_type=args.band_fusion_type,
        tf_fusion_type='concat_linear',
        global_fusion_type='concat_linear',
        hyperbolic_depth=args.hyperbolic_depth,
        hyperbolic_curvature=args.hyperbolic_curvature,
        learnable_curvature=not args.fixed_curvature,
        tf_align_weight=args.tf_align_weight,
        tf_align_type=args.tf_align_type,
        raw_mask_weight=args.raw_mask_weight,
        raw_mask_type=args.raw_mask_type,
    )

    if args.save_every is not None:
        unit = 'epoch' if args.epochs is not None else 'iter'
        config[f'after_{unit}_callback'] = save_checkpoint_callback(args.save_every, unit)

    run_dir_name = name_with_datetime(args.run_name) if is_main else None
    run_dir_name = broadcast_object(run_dir_name)
    run_dir = 'training/' + args.dataset + '__' + run_dir_name
    if is_main:
        os.makedirs(run_dir, exist_ok=True)
    sync_barrier()

    t = time.time()
    model = TS2VecThreeBranchHypRelationCGeom(
        input_dims=train_data.shape[-1],
        device=device,
        distributed=distributed,
        **config,
    )
    model.fit(train_data, n_epochs=args.epochs, n_iters=args.iters, verbose=is_main)
    sync_barrier()
    if is_main:
        model.save(f'{run_dir}/model.pkl')
        pkl_save(f'{run_dir}/model_config.pkl', config)
        pkl_save(f'{run_dir}/train_history.pkl', model.train_history)

    t = time.time() - t
    if is_main:
        print(f'\nTraining time: {datetime.timedelta(seconds=t)}\n')

    if args.eval and is_main:
        _, eval_res = tasks.eval_classification(
            model,
            train_data,
            train_labels,
            test_data,
            test_labels,
            eval_protocol='linear',
        )
        eval_res['model_config'] = config
        pkl_save(f'{run_dir}/eval_res.pkl', eval_res)
        print('Evaluation result:', eval_res)

    sync_barrier()
    cleanup_distributed()
    if is_main:
        print('Finished.')
