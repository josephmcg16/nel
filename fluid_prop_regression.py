import pandas as pd
import yaml
from fluid_properties.regression import DataLoader, train, plot_results


if __name__ == '__main__':
    with open('config.yaml') as config_file:
        CONFIG = yaml.load(config_file, Loader=yaml.FullLoader)

    loader = DataLoader(CONFIG)
    predictor, df_errors, df_coeffecients = train(loader, CONFIG)
    predictor.fit(loader.X_train, loader.Y_train)

    df_coeffecients.mean().to_csv(
        f'{CONFIG["output_dir"]}/{CONFIG["FLUID"]}_coefficients.csv', index_label='Feature Name', header=['Coeffecient']
    )
    df_errors.to_csv(
        f'{CONFIG["output_dir"]}/{CONFIG["FLUID"]}_regression_errors.csv', index_label='Split Number'
    )

    df_test = pd.DataFrame()
    df_test[CONFIG['x_feature']], df_test[CONFIG['y_feature']] = loader.X_test.T
    df_test[CONFIG['z_feature']] = loader.Y_test
    df_test[f'Predicted {CONFIG["z_feature"]}'] = predictor.predict(loader.X_test)
    df_test[CONFIG["error_feature"]] = (df_test[f'Predicted {CONFIG["z_feature"]}'] - df_test[CONFIG["z_feature"]]).abs()

    plot_results(df_test, df_errors, CONFIG)
