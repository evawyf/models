


"""Example experiment configuration definition."""
from typing import List

import dataclasses

from official.core import config_definitions as cfg
from official.core import exp_factory
from official.modeling import hyperparams
from official.modeling import optimization


@dataclasses.dataclass
class DataConfig(cfg.DataConfig):
    """Input config for training. Add more fields as needed."""
#   output_size: List[int] = dataclasses.field(default_factory=list)
#   # If crop_size is specified, image will be resized first to
#   # output_size, then crop of size crop_size will be cropped.
#   crop_size: List[int] = dataclasses.field(default_factory=list)
    input_path: str = ''
    global_batch_size: int = 0
    is_training: bool = True
    dtype: str = 'float32'
    shuffle_buffer_size: int = 10000
    cycle_length: int = 10
    file_type: str = 'tfrecord'


@dataclasses.dataclass
class YOLOv3Model(hyperparams.Config):
    """The model config. Used by build_example_model function."""
    n_classes: int = 0
    input_size: List[int] = dataclasses.field(default_factory=list)
    use_bias: bool = False
    # common
    # norm_activation: common.NormActivation = common.NormActivation()


@dataclasses.dataclass
class Losses(hyperparams.Config):
    l2_weight_decay: float = 0.0


@dataclasses.dataclass
class Evaluation(hyperparams.Config):
    top_k: int = 5


@dataclasses.dataclass
class YOLOv3Task(cfg.TaskConfig):
    """The task config."""
    model: YOLOv3Model = YOLOv3Model()
    train_data: DataConfig = DataConfig(is_training=True)
    validation_data: DataConfig = DataConfig(is_training=False)
    losses: Losses = Losses()
    # evaluation: Evaluation = Evaluation()


@exp_factory.register_config_factory('yolo3_experiment')
def yolo3_experiment(): # -> cfg.ExperimentConfig:
    """Definition of a full example experiment."""
    train_batch_size = 256
    eval_batch_size = 256
    steps_per_epoch = 10
    config = cfg.ExperimentConfig(
        task=YOLOv3Task(
            model=YOLOv3Model(num_classes=80, input_size=[128, 128, 3]),
            losses=Losses(l2_weight_decay=1e-4),
            train_data=DataConfig(
                input_path='/path/to/train*',
                is_training=True,
                global_batch_size=train_batch_size),
            validation_data=DataConfig(
                input_path='/path/to/valid*',
                is_training=False,
                global_batch_size=eval_batch_size)),
        trainer=cfg.TrainerConfig(
            steps_per_loop=steps_per_epoch,
            summary_interval=steps_per_epoch,
            checkpoint_interval=steps_per_epoch,
            train_steps=90 * steps_per_epoch,
            validation_steps=steps_per_epoch,
            validation_interval=steps_per_epoch,
            optimizer_config=optimization.OptimizationConfig({
                'optimizer': {
                    'type': 'adam',
                    'sgd': {
                        'momentum': 0.9
                    }
                },
                'learning_rate': {
                    'type': 'cosine',
                    'cosine': {
                        'initial_learning_rate': 1.6,
                        'decay_steps': 350 * steps_per_epoch
                    }
                },
                'warmup': {
                    'type': 'linear',
                    'linear': {
                        'warmup_steps': 5 * steps_per_epoch,
                        'warmup_learning_rate': 0
                    }
                }
            })),
        restrictions=[
            'task.train_data.is_training != None',
            'task.validation_data.is_training != None'
        ])

    return config