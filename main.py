from data_loading import DataLoader
from preprocessing import Preprocessing
from feature_selection import FeatureSelection

def main():
    # Load data
    loader = DataLoader('data.csv', 'json_file.json')
    data = loader.load_data()
    json_data = loader.load_json()

    # Preprocess data
    prep = Preprocessing(data)
    prep.convert_to_factor(['col1', 'col2'])
    prep.filter_data([prep.data['col3'] > 0])
    
    # Split data into waves
    waves = prep.data['Ola'].unique()
    wave_data = {wave: prep.data[prep.data['Ola'] == wave] for wave in waves}

    for wave, data in wave_data.items():
        # Split data into train and test
        train, test = prep.split_data(data)

        # Perform feature selection
        fs = FeatureSelection(train, 'target')
        importance = fs.calculate_importance('rf')
        selected_features = fs.select_features(importance)
        averaged_importance = fs.averaged_importance('rf')

        # Save results
        pd.DataFrame(selected_features).to_csv(f'selected_features_wave_{wave}.csv')
        with open(f'averaged_importance_wave_{wave}.txt', 'w') as f:
            f.write(str(averaged_importance))

if __name__ == "__main__":
    main()
