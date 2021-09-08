
"""YOLOv3 task definition."""
from typing import Optional, Any, Tuple, List

from absl import logging
import tensorflow as tf

from official.common import dataset_fn
from official.core import base_task
from official.core import input_reader
from official.core import task_factory

from official.projects.yolo3.configs import yolo3_cfg as exp_cfg
from official.projects.yolo3.modeling import yolo3_model
from official.projects.yolo3.loss import yolo3_loss
# from official.projects.yolo3.ops import metrics as yolo3_metrics

from official.vision.beta.dataloaders import segmentation_input


@task_factory.register_task_cls(exp_cfg.YOLOv3Task)
class YOLOv3Task(base_task.Task):
    """Class of an example task.
    A task is a subclass of base_task.Task that defines model, input, loss, metric
    and one training and evaluation step, etc.
    """

    def build_model(self): # -> tf.keras.Model:
        """Builds a model."""
        input_specs = tf.keras.layers.InputSpec(
            shape=[None] + self.task_config.model.input_size)

        model = yolo3_model.build_yolo3_model(
            input_specs=input_specs, model_config=self.task_config.model)
        return model


    def build_inputs(
        self,
        params: exp_cfg.DataConfig,
        input_context: Optional[tf.distribute.InputContext] = None):
        """Builds YOLOv3 input."""

        ignore_label = self.task_config.losses.ignore_label

        decoder = segmentation_input.Decoder()
        parser = segmentation_input.Parser(
            output_size=params.output_size,
            crop_size=params.crop_size,
            ignore_label=ignore_label,
            aug_rand_hflip=params.aug_rand_hflip,
            dtype=params.dtype)

        reader = input_reader.InputReader(
            params,
            dataset_fn=dataset_fn.pick_dataset_fn(params.file_type),
            decoder_fn=decoder.decode,
            parser_fn=parser.parse_fn(params.is_training))

        dataset = reader.read(input_context=input_context)

        return dataset


    def build_losses(self,
                    labels: tf.Tensor,
                    model_outputs: tf.Tensor,
                    aux_losses: Optional[Any] = None): # -> tf.Tensor:
        """Builds losses for training and validation.
        Args:
        labels: Input groundtruth labels.
        model_outputs: Output of the model.
        aux_losses: The auxiliarly loss tensors, i.e. `losses` in tf.keras.Model.
        Returns:
        The total loss tensor.
        """
        # total_loss = tf.keras.losses.sparse_categorical_crossentropy(
        #     labels, model_outputs, from_logits=True)
        # total_loss = tf_utils.safe_mean(total_loss)

        total_loss = yolo3_losses.YOLOv3Loss(model_outputs, labels['masks'])

        if aux_losses:
            total_loss += tf.add_n(aux_losses)

        return total_loss


    # TODO: 
    def build_metrics(self,
        training: bool = True): # -> Sequence[tf.keras.metrics.Metric]:
        """Gets streaming metrics for training/validation.
        This function builds and returns a list of metrics to compute during
        training and validation. The list contains objects of subclasses of
        tf.keras.metrics.Metric. Training and validation can have different metrics.
        Args:
        training: Whether the metric is for training or not.
        Returns:
        A list of tf.keras.metrics.Metric objects.
        """
        k = self.task_config.evaluation.top_k
        metrics = [
            tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(
                k=k, name='top_{}_accuracy'.format(k))
        ]
        return metrics



    def train_step(self,
                    inputs: Tuple[Any, Any],
                    model: tf.keras.Model,
                    optimizer: tf.keras.optimizers.Optimizer,
                    metrics: Optional[List[Any]] = None): # -> Mapping[str, Any]:
        """Does forward and backward.
        This example assumes input is a tuple of (features, labels), which follows
        the output from data loader, i.e., Parser. The output from Parser is fed
        into train_step to perform one step forward and backward pass. Other data
        structure, such as dictionary, can also be used, as long as it is consistent
        between output from Parser and input used here.
        Args:
        inputs: A tuple of of input tensors of (features, labels).
        model: A tf.keras.Model instance.
        optimizer: The optimizer for this training step.
        metrics: A nested structure of metrics objects.
        Returns:
        A dictionary of logs.
        """
        features, labels = inputs
        num_replicas = tf.distribute.get_strategy().num_replicas_in_sync
        with tf.GradientTape() as tape:
            outputs = model(features, training=True)
            # Casting output layer as float32 is necessary when mixed_precision is
            # mixed_float16 or mixed_bfloat16 to ensure output is casted as float32.
            outputs = tf.nest.map_structure(lambda x: tf.cast(x, tf.float32), outputs)

            # Computes per-replica loss.
            loss = self.build_losses(
                model_outputs=outputs, labels=labels, aux_losses=model.losses)
            # Scales loss as the default gradients allreduce performs sum inside the
            # optimizer.
            scaled_loss = loss / num_replicas

            # For mixed_precision policy, when LossScaleOptimizer is used, loss is
            # scaled for numerical stability.
            if isinstance(optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
                scaled_loss = optimizer.get_scaled_loss(scaled_loss)

        tvars = model.trainable_variables
        grads = tape.gradient(scaled_loss, tvars)
        # Scales back gradient before apply_gradients when LossScaleOptimizer is
        # used.
        if isinstance(optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
            grads = optimizer.get_unscaled_gradients(grads)
        optimizer.apply_gradients(list(zip(grads, tvars)))

        logs = {self.loss: loss}
        if metrics:
            self.process_metrics(metrics, labels, outputs)
        return logs

    def validation_step(self,
                        inputs: Tuple[Any, Any],
                        model: tf.keras.Model,
                        metrics: Optional[List[Any]] = None): # -> Mapping[str, Any]:
        """Runs validatation step.
        Args:
        inputs: A tuple of of input tensors of (features, labels).
        model: A tf.keras.Model instance.
        metrics: A nested structure of metrics objects.
        Returns:
        A dictionary of logs.
        """
        features, labels = inputs
        outputs = self.inference_step(features, model)
        outputs = tf.nest.map_structure(lambda x: tf.cast(x, tf.float32), outputs)
        loss = self.build_losses(
            model_outputs=outputs, labels=labels, aux_losses=model.losses)

        logs = {self.loss: loss}
        if metrics:
            self.process_metrics(metrics, labels, outputs)
        return logs

    def inference_step(self, inputs: tf.Tensor, model: tf.keras.Model): # -> Any:
        """Performs the forward step. It is used in validation_step."""
        return model(inputs, training=False)