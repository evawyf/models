

"""TensorFlow Model Garden Vision training driver."""

from absl import app

# pylint: disable=unused-import
from official.common import flags as tfm_flags
from official.projects.basnet.configs import yolo3 as yolo3_cfg
# from official.projects.basnet.modeling import yolo3_model
# from official.projects.basnet.tasks import basnet as basenet_task
from official.vision.beta import train


if __name__ == '__main__':
  tfm_flags.define_flags()
  app.run(train.main)