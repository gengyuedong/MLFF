import os

# Ensure tf doesnt use more cpus than requested in job
n = 1
os.environ["OMP_NUM_THREADS"] = str(n)
os.environ['TF_NUM_INTEROP_THREADS'] = str(n)
os.environ['TF_NUM_INTRAOP_THREADS'] = str(n)

import pickle
import random
from datetime import datetime

from pymatgen.core.structure import Structure

from m3gnet.models import M3GNet, Potential
from m3gnet.trainers import PotentialTrainer
from m3gnet.layers._atom_ref import AtomRef

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
tf.config.optimizer.set_jit(True) # faster

from monty.serialization import loadfn

# Learning rate
initial_learning_rate = 5*1e-4
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=10000,
    decay_rate=0.96,
    staircase=True
)

# Load Element Reference energies
config = loadfn("./atom_ref.json")
# print(config)
element_refs = config

# Initialise model
m3gnet = M3GNet(
    is_intensive=False,
    element_refs=element_refs,
)
print("Element refs:", m3gnet.element_ref_calc.property_per_element)
# Load pretrained model
m3gnet = m3gnet.load(model_name='MP-2021.2.8-EFS')
# Reset element_ref
m3gnet.set_element_refs(element_refs=element_refs)

potential = Potential(model=m3gnet)
trainer = PotentialTrainer(
    potential=potential,
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule)
)

# Data files
train_file = "train_dic.pkl"
val_file = "val_dic.pkl"
# Load train data
with open(train_file, "rb") as f:
    train_data = pickle.load(f)
# Load validation data
with open(val_file, "rb") as f:
    val_data = pickle.load(f)
print("Shape of training inputs")
for key in train_data.keys():
    print(key, len(train_data[key]))
print("Shape of validation inputs")
for key in val_data.keys():
    print(key, len(val_data[key]))

# Train
batch_size = 4
epochs = 30
print(f"Batch size: {batch_size}")
trainer.train(
    graphs_or_structures=train_data["structure"],
    energies=train_data["energy"],
    forces=train_data["forces"],
    stresses=train_data["stress"],
    validation_graphs_or_structures=val_data["structure"],
    val_energies=val_data["energy"],
    val_forces=val_data["forces"],
    val_stresses=val_data["stress"],
    epochs=epochs,
    fit_per_element_offset=False, # Using HSE06 bulk energies
    save_checkpoint=True,
    batch_size=batch_size,
)

date = datetime.now().strftime("%Y_%m_%d_%H_%M")
name = f"m3gnet_{date}_{epochs}_epochs"
m3gnet.save(dirname=name)
