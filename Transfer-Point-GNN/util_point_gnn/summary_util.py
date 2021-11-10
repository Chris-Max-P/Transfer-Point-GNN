"""This file implements utility functions for tensorflow summary."""

import tensorflow as tf
from tensorboard import summary as summary_lib

def write_summary_scale(key, value, global_step, summary_dir):
    """Write a scale summary to summary_dir. """
    writer = tf.compat.v1.summary.FileWriterCache.get(summary_dir)
    summary = tf.compat.v1.Summary(value=[
        tf.compat.v1.Summary.Value(tag=key, simple_value=value),
    ])
    writer.add_summary(summary, global_step)

def write_summary(summary, global_step, summary_dir):
    """Write a summary to summary_dir. """
    writer = tf.compat.v1.summary.FileWriterCache.get(summary_dir)
    writer.add_summary(summary, global_step)
