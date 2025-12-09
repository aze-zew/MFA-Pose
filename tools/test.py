# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

import time
import torch

import mmengine
from mmengine.config import Config, DictAction
from mmengine.hooks import Hook
from mmengine.runner import Runner


# 新增的FPS测量Hook类
class FPSMetricHook(Hook):
    """Hook to measure FPS during testing."""
    
    def __init__(self, warmup_iters=10):
        self.warmup_iters = warmup_iters
        self.start_event = None
        self.end_event = None
        self.iter_times = []
        self.frame_counts = []
    
    def before_test(self, runner):
        """Initialize timing events and reset counters."""
        # 只在主进程进行测量
        if runner.rank != 0:
            return
            
        if torch.cuda.is_available():
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
        
        self.iter_times = []
        self.frame_counts = []
        self.warmup_counter = 0
    
    def before_test_iter(self, runner, batch_idx, data_batch=None):
        """Record start time before each iteration."""
        if runner.rank != 0 or self.warmup_counter < self.warmup_iters:
            return
            
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.start_event.record()
        else:
            self.iter_start_time = time.time()
    
    def after_test_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        """Record end time after each iteration and calculate iteration time."""
        if runner.rank != 0:
            return
            
        # 跳过预热迭代
        if self.warmup_counter < self.warmup_iters:
            self.warmup_counter += 1
            return
        
        # 记录结束时间并计算迭代耗时
        if torch.cuda.is_available():
            self.end_event.record()
            torch.cuda.synchronize()
            iter_time = self.start_event.elapsed_time(self.end_event) / 1000.0  # 毫秒转秒
        else:
            iter_time = time.time() - self.iter_start_time
        
        # 获取实际处理的帧数（考虑可能的batch size变化）
        if hasattr(runner.test_loop.dataloader, 'batch_size'):
            batch_size = runner.test_loop.dataloader.batch_size
        elif 'inputs' in data_batch:
            batch_size = len(data_batch['inputs'])
        else:
            batch_size = 1
        
        self.iter_times.append(iter_time)
        self.frame_counts.append(batch_size)
    
    def after_test(self, runner):
        """Calculate and log FPS after testing."""
        if runner.rank != 0 or not self.iter_times:
            return
            
        total_frames = sum(self.frame_counts)
        total_time = sum(self.iter_times)
        
        if total_time > 0:
            fps = total_frames / total_time
            avg_time_per_frame = total_time / total_frames
            
            # 输出FPS结果到控制台
            runner.logger.info(f'Test FPS: {fps:.2f} frames/s')
            runner.logger.info(f'Average time per frame: {avg_time_per_frame*1000:.2f} ms')
            
            # 可选：保存FPS结果到文件
            result_path = osp.join(runner.work_dir, 'fps_results.json')
            fps_data = {
                'total_frames': total_frames,
                'total_time': total_time,
                'fps': fps,
                'avg_time_per_frame': avg_time_per_frame,
                'hardware': {
                    'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
                    'cuda_version': torch.version.cuda if torch.cuda.is_available() else 'N/A'
                },
                'batch_size': runner.test_loop.dataloader.batch_size,
                'warmup_iters': self.warmup_iters,
                'measured_iters': len(self.iter_times)
            }
            mmengine.dump(fps_data, result_path)
            runner.logger.info(f'FPS results saved to {result_path}')
        else:
            runner.logger.warning('FPS measurement failed: total_time is zero')




def parse_args():
    parser = argparse.ArgumentParser(
        description='MMPose test (and eval) model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir', help='the directory to save evaluation results')
    parser.add_argument('--out', help='the file to save metric results.')
    parser.add_argument(
        '--dump',
        type=str,
        help='dump predictions to a pickle file for offline evaluation')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    parser.add_argument(
        '--show-dir',
        help='directory where the visualization images will be saved.')
    parser.add_argument(
        '--show',
        action='store_true',
        help='whether to display the prediction results in a window.')
    parser.add_argument(
        '--interval',
        type=int,
        default=1,
        help='visualize per interval samples.')
    parser.add_argument(
        '--wait-time',
        type=float,
        default=1,
        help='display time of every window. (second)')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')

    # 新增FPS测量选项
    parser.add_argument(
        '--fps',
        action='store_true',
        help='enable FPS measurement during testing')

    
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/test.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    parser.add_argument(
        '--badcase',
        action='store_true',
        help='whether analyze badcase in test')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def merge_args(cfg, args):
    """Merge CLI arguments to config."""

    cfg.launcher = args.launcher
    cfg.load_from = args.checkpoint

    # -------------------- work directory --------------------
    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # -------------------- visualization --------------------
    if (args.show and not args.badcase) or (args.show_dir is not None):
        assert 'visualization' in cfg.default_hooks, \
            'PoseVisualizationHook is not set in the ' \
            '`default_hooks` field of config. Please set ' \
            '`visualization=dict(type="PoseVisualizationHook")`'

        cfg.default_hooks.visualization.enable = True
        cfg.default_hooks.visualization.show = False \
            if args.badcase else args.show
        if args.show:
            cfg.default_hooks.visualization.wait_time = args.wait_time
        cfg.default_hooks.visualization.out_dir = args.show_dir
        cfg.default_hooks.visualization.interval = args.interval

    # -------------------- badcase analyze --------------------
    if args.badcase:
        assert 'badcase' in cfg.default_hooks, \
            'BadcaseAnalyzeHook is not set in the ' \
            '`default_hooks` field of config. Please set ' \
            '`badcase=dict(type="BadcaseAnalyzeHook")`'

        cfg.default_hooks.badcase.enable = True
        cfg.default_hooks.badcase.show = args.show
        if args.show:
            cfg.default_hooks.badcase.wait_time = args.wait_time
        cfg.default_hooks.badcase.interval = args.interval

        metric_type = cfg.default_hooks.badcase.get('metric_type', 'loss')
        if metric_type not in ['loss', 'accuracy']:
            raise ValueError('Only support badcase metric type'
                             "in ['loss', 'accuracy']")

        if metric_type == 'loss':
            if not cfg.default_hooks.badcase.get('metric'):
                cfg.default_hooks.badcase.metric = cfg.model.head.loss
        else:
            if not cfg.default_hooks.badcase.get('metric'):
                cfg.default_hooks.badcase.metric = cfg.test_evaluator

    # -------------------- Dump predictions --------------------
    if args.dump is not None:
        assert args.dump.endswith(('.pkl', '.pickle')), \
            'The dump file must be a pkl file.'
        dump_metric = dict(type='DumpResults', out_file_path=args.dump)
        if isinstance(cfg.test_evaluator, (list, tuple)):
            cfg.test_evaluator = [*cfg.test_evaluator, dump_metric]
        else:
            cfg.test_evaluator = [cfg.test_evaluator, dump_metric]

    # -------------------- Other arguments --------------------
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    return cfg


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    cfg = merge_args(cfg, args)

    # build the runner from config
    runner = Runner.from_cfg(cfg)

    if args.out:

        class SaveMetricHook(Hook):

            def after_test_epoch(self, _, metrics=None):
                if metrics is not None:
                    mmengine.dump(metrics, args.out)

        runner.register_hook(SaveMetricHook(), 'LOWEST')

    # 新增：注册FPS测量Hook
    if args.fps:
        fps_hook = FPSMetricHook(warmup_iters=10)
        runner.register_hook(fps_hook)

    # start testing
    runner.test()


if __name__ == '__main__':
    main()
