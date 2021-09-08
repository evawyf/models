

"""TensorFlow Model Garden Vision training driver."""

from absl import app

# pylint: disable=unused-import
from official.common import flags as tfm_flags
from official.projects.yolo3.configs import yolo3_cfg
from official.projects.yolo3.modeling import yolo3_model
from official.projects.yolo3.tasks import yolo3_task
from official.vision.beta import train


if __name__ == '__main__':
  tfm_flags.define_flags()
  app.run(train.main)