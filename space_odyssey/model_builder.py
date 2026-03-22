import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.readwrite import BIFWriter

df = pd.read_csv("data.csv", index_col=0)
# pgmpy expects discrete/categorical columns for MLE in this model.
df = df.astype("category")

model = DiscreteBayesianNetwork([
    ('HAL_Switch', 'HAL'),
    ('System_Age', 'Thermostat'),
    ('System_Age', 'O2_Generator'),
    ('System_Age', 'CO2_Scrubber'),
    ('System_Age',  'Diagnosis'),
    ('HAL', 'AI_Test'),
    ('HAL', 'Manoeuvre'),
    ('HAL', 'Thermostat'),
    ('HAL', 'O2_Generator'),
    ('HAL', 'CO2_Scrubber'),
    ('Manoeuvre', 'Alien_Attack'),
    ('Manoeuvre', 'Meteor_Shower'),
    ('Thermostat', 'Temperature'),
    ('O2_Generator', 'O2_Level'),
    ('CO2_Scrubber', 'CO2_Level'),
    ('Alien_Attack', 'Crew_Status'),
    ('Alien_Attack', 'Alert_System'),
    ('Alien_Attack', 'Decompression'),
    ('Alien_Attack', 'Porthole'),
    ('Meteor_Shower', 'Alert_System'),
    ('Meteor_Shower', 'Decompression'),
    ('Meteor_Shower', 'Porthole'),
    ('Decompression', 'Temperature'),
    ('Decompression', 'O2_Level'),
    ('Decompression', 'CO2_Level'),
    ('Temperature', 'Life_Support'),
    ('O2_Level', 'Life_Support'),
    ('CO2_Level', 'Life_Support'),
    ('Life_Support', 'Crew_Status')
])

model.fit(df, estimator=MaximumLikelihoodEstimator)
writer = BIFWriter(model)
writer.write_bif(filename="spaceship.bif")
print('spaceship.bif file has been created successfully.')