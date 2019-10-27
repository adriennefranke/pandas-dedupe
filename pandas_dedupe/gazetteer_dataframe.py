from pandas_dedupe.utility_functions import *

import dedupe

import pandas as pd

def gazetteer_dataframes(messy_df, canonical_df, field_properties, recall_weight, n_matches, config_name="gazetteer_dataframes"):

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
            gazetteer = dedupe.StaticGazetteer(sf)

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

        # If we have training data saved from a previous run of linker,
        # look for it an load it in.
        # __Note:__ if you want to train from scratch, delete the training_file
        if os.path.exists(training_file):
            print('reading labeled examples from ', training_file)
            with open(training_file) as tf :
                linker.readTraining(tf)

        # ## Active learning
        # Dedupe will find the next pair of records
        # it is least certain about and ask you to label them as matches
        # or not.
        # use 'y', 'n' and 'u' keys to flag duplicates
        # press 'f' when you are finished
        print('starting active labeling...')


        dedupe.consoleLabel(gazetteer)

        gazetteer.train()

        # When finished, save our training awak to disk

        with open(training_file, 'w') as tf:
            gazetteer.writeTraining(tf)

        # Make the canonical set
        gazetteer.index(data_2)

        # Save our weights and predicates to disk. If the settings file exists,
        # we will skip all training and learning next time we run this file.
        with open(settings_file, 'wb') as sf:
            gazetteer.writeSettings(sf, index=True)

        gazetteer.cleanupTraining()

    gazetteer.index(data_2)
    #Calc threshold
    print('start calculating threshold')
    threshold = gazetteer.threshold(data_1, recall_weight)
    print('Threshold: {}'.format(threshold))
    

    results = gazetteer.match(data_1, threshold=threshold, n_matches=n_matches)

    results_df = pd.DataFrame(results)

    results_df['messy_df_link'] = results_df[0].apply(lambda x: x[0][0])
    results_df['messy_df_link'] = results_df['messy_df_link'].str.strip('messy_df')
    results_df['messy_df_link'] = results_df['messy_df_link'].astype(int)

    results_df['canonical_df_link'] = results_df[0].apply(lambda x: x[0][1])
    results_df['canonical_df_link'] = results_df['canonical_df_link'].str.strip('canonical_df')
    results_df['canonical_df_link'] = results_df['canonical_df_link'].astype(int)

    results_df['confidence'] = results_df[0].apply(lambda x: x[1])
    results_df['cluster id'] = results_df.index

    results_df = results_df.rename(columns={0: 'results'})
    results_df['results'] = results_df['results'].astype(str)

    #For both messy_df & canonical_df, add cluster id & confidence score from results_df
    messy_df.index.rename('messy_df_link', inplace=True)
    messy_df = messy_df.rename(columns={'unique_id': 'messy_unique_id'})
    messy_df = messy_df.merge(results_df_copy, on='messy_df_link', how='left')

    canonical_df.index.rename('canonical_df_link', inplace=True)
    canonical_df = canonical_df.rename(columns={'unique_id': 'canonical_unique_id'})
    canonical_df = canonical_df.merge(results_df_copy, on='canonical_df_link', how='left')

    #Merge messy_df & canonical_df together
    final_df = messy_df.merge(canonical_df, on='results')


    return final_df


