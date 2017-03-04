import tensorflow as tf
import os
from tensorflow.python.util import compat

x = tf.placeholder(tf.float32, shape=(None))
y = tf.placeholder(tf.float32, shape=(None))

three = tf.Variable(3, dtype= tf.float32)
z = tf.scalar_mul(three, x) + y

model_version = 1
path = os.path.join("three_x_plus_y", str(model_version))
builder = tf.python.saved_model.builder.SavedModelBuilder(path)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    builder.add_meta_graph_and_variables(
        sess,
        [tf.python.saved_model.tag_constants.SERVING],
        signature_def_map= {
            "magic_model": tf.saved_model.signature_def_utils.predict_signature_def(
                inputs= {"egg": x, "bacon": y},
                outputs= {"spam": z})
        })
    builder.save()
