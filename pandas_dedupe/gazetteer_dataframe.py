from pandas_dedupe.utility_functions import *

import dedupe

import pandas as pd

def gazetteer_dataframes(messy_df, canonical_df, field_properties, config_name="gazetteer_dataframes"):

	config_name = config_name.replace(" ", "_")

    settings_file = config_name + '_learned_settings'
    training_file = config_name + '_training.json'
 
    print('importing data ...')

    messy_df = clean_punctuation(messy_df)
    specify_type(messy_df, field_properties)

    messy_df['index_field'] = messy_df.index
    messy_df['index_field'] = messy_df['index_field'].apply(lambda x: "messy_df" + str(x))
    messy_df.set_index(['index_field'], inplace=True)
            
    data_1 = messy_df.to_dict(orient='index')


	canonical_df = clean_punctuation(canonical_df)
    specify_type(canonical_df, field_properties)
    
    canonical_df['index_field'] = canonical_df.index
    canonical_df['index_field'] = canonical_df['index_field'].apply(lambda x: "canonical_df" + str(x))
    canonical_df.set_index(['index_field'], inplace=True)

    data_2 = canonical_df.to_dict(orient='index')

	# ---------------------------------------------------------------------------------



    # ## Training


    if os.path.exists(settings_file):
        print('reading from', settings_file)
        with open(settings_file, 'rb') as sf :
            linker = dedupe.StaticGazetteer(sf)

    else:
        # Define the fields the linker will pay attention to
        #
        # Notice how we are telling the linker to use a custom field comparator
        # for the 'price' field. 

        fields = []
        select_fields(fields, field_properties)
                
              
                
        # Create a new gazetteer object and pass our data model to it.
        gazetteer = dedupe.Gazetteer(fields)
        
        # To train the gazetteer, we feed it a sample of records.
        gazetteer.sample(data_1, data_2, 15000)





