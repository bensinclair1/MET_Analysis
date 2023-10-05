from datetime import datetime
import pandas as pd
import numpy as np
import requests
import os

# Setting up timing mappings to translate MATLAB syntax
TIMING_MAPPING = {
    "hourly": "H",
    "daily": "D",
}

REVERSE_TIMING_MAPPING = {v: k for k, v in TIMING_MAPPING.items()}

class CTDataProcessor:
    
    def __init__(self, timing="hourly"):
        self.timing = TIMING_MAPPING.get(timing, timing)

    def get_human_readable_timing(self):
        return REVERSE_TIMING_MAPPING.get(self.timing, self.timing)


    def process(self, filename="Results.csv"):

        # Read CT Data
        df = self.read_CT_data(filename)
        
        # get the list of deviceName
        device_names = self.list_machines(df)
        
        # Remove Power Failure detected rows
        df = df[df['powerFailureDetected'] != 1]

        # Export data for each machine
        for device in device_names:
            # Get current data
            temp_df = df[df['deviceName'] == device][["A", "channel1", "channel2", "channel3"]].dropna()
            
            # Aggregate data
            temp_df = temp_df.resample(self.timing).mean().fillna(0)

            # Placeholder columns
            temp_df["V"] = 0
            temp_df["kW"] = 0
            temp_df["cost"] = 0
            temp_df["neutral"] = 0
            
            directory_name = str(device)
            if not os.path.exists(directory_name):
                os.makedirs(directory_name)
                
            file_name = os.path.join(directory_name, f"{self.get_human_readable_timing()}_{device}.csv")
            temp_df.to_csv(file_name)

    def read_CT_data(self, filename):
        df = pd.read_csv(filename)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        print(df.head())
        return df.dropna()

    def list_machines(self, df):
        devices = df['deviceName'].unique()
        print(devices)
        return devices
    
class CTDataAnalyser:
    
    def __init__(self, MACHINE_NAMES, TIMING, PHASE, VOLTAGE, DAY_UNIT_COST, NIGHT_UNIT_COST, NIGHT_TARIFF_START_TIME, NIGHT_TARIFF_END_TIME, REGION):
        self.MACHINE_NAMES = MACHINE_NAMES
        self.TIMING = TIMING
        self.PHASE = PHASE
        self.VOLTAGE = VOLTAGE
        self.DAY_UNIT_COST = DAY_UNIT_COST
        self.NIGHT_UNIT_COST = NIGHT_UNIT_COST
        self.NIGHT_TARIFF_START_TIME = NIGHT_TARIFF_START_TIME
        self.NIGHT_TARIFF_END_TIME = NIGHT_TARIFF_END_TIME
        self.REGION = REGION

    def machine_calculations(self):
        
        # Getting adjusted timescale
        self.tscale = self.adjust_timescale()

        for device_name in self.MACHINE_NAMES:
            
            # Loading dataframe of individual machine
            device_df = pd.read_csv(f'{device_name}/{self.TIMING}_{device_name}.csv')

            device_df = self.compute_kW_utilization_and_cost(device_df, device_name)

            device_df = self.estimate_load_imbalance(device_df, device_name)

            # Carbon emission kgCO2/kWh
            if self.TIMING.lower() != 'minutely':
                if self.REGION.lower() == 'national':
                    device_df = self.estimate_carbon_emissions(device_df)

            device_df.to_csv(f'{device_name}/{self.TIMING}_{device_name}.csv', index=False)

    # Calculates deviation from average between current channels for each individual current channel
    def unbalanced(self, channels):

        # Convert to numpy array for easier calculations
        channels = np.array(channels).T # Transpose to get channels as columns

        Iav = np.mean(channels, axis=1)

        u = np.max(np.abs(channels - Iav[:, np.newaxis]), axis=1) / Iav * 100
        u[np.isnan(u)] = 0
        
        return u
    
    def adjust_timescale(self):
        if self.TIMING == 'hourly':
            tscale = 1
        elif self.TIMING == 'minutely':
            tscale = 60
        elif self.TIMING == 'daily':
            tscale = 1/24
        elif self.TIMING == 'weekly':
            tscale = 1/(24*7)
        else:
            raise ValueError("Invalid timing value")
        return tscale
    
    def compute_kW_utilization_and_cost(self, device_df, device_name):
        # Only 3-phase Machines' Current should be multiplied by sqrt(3)
        device_df['A'] = device_df['A'] * (3 ** 0.5 if self.PHASE[self.MACHINE_NAMES.index(device_name)] == 3 else 1) / self.tscale

        # Insert the voltages of individual machines
        device_df['V'] = self.VOLTAGE[self.MACHINE_NAMES.index(device_name)]

        # Estimate kW
        device_df['kW'] = device_df['V'] * device_df['A'] / 1000

        # Estimate Utilization
        device_df['utilization'] = (device_df['A'] > 1).astype(int)

        # Estimate unit cost of electricity
        device_df['p'] = self.DAY_UNIT_COST[self.MACHINE_NAMES.index(device_name)]
        device_df['timestamp'] = pd.to_datetime(device_df['timestamp'])
        device_df.loc[device_df['timestamp'].dt.hour >= self.NIGHT_TARIFF_START_TIME, 'p'] = self.NIGHT_UNIT_COST[self.MACHINE_NAMES.index(device_name)]
        device_df.loc[device_df['timestamp'].dt.hour < self.NIGHT_TARIFF_END_TIME, 'p'] = self.NIGHT_UNIT_COST[self.MACHINE_NAMES.index(device_name)]
        
        device_df['cost'] = device_df['kW'] * device_df['p']
        return device_df
    
    def estimate_load_imbalance(self, device_df, device_name):
        if self.PHASE[self.MACHINE_NAMES.index(device_name)] == 1:
            device_df['unbalanced'] = self.unbalanced([device_df['channel1']])
        elif self.PHASE[self.MACHINE_NAMES.index(device_name)] == 3:
            device_df['unbalanced'] = self.unbalanced([device_df['channel1'], device_df['channel2'], device_df['channel3']])
        return device_df
    
    def estimate_carbon_emissions(self, device_df):
        st = str(device_df['timestamp'].iloc[0])
        en = str(device_df['timestamp'].iloc[-1])

        timestamp, co2 = self.fetch_carbon_emission_data(st, en)
        
        # Creating a DataFrame from the retrieved data
        tco2 = pd.DataFrame({'timestamp': timestamp, 'co2': co2})
        tco2['timestamp'] = pd.to_datetime(tco2['timestamp'])

        # Re-sampling operations
        # The MATLAB code was using 'retime' to handle the time series data.
        # `resample` and `interpolate` achieve a similar result.
        tco2.set_index('timestamp', inplace=True)
        tco2 = tco2.resample(TIMING_MAPPING.get(self.TIMING, self.TIMING)).mean().interpolate().reset_index()
        
        # Merging with the original DataFrame M
        device_df = pd.merge(device_df, tco2, on='timestamp', how='left')
        device_df['co2'].fillna(0, inplace=True)
        device_df['co2'] = device_df['kW'] * device_df['co2']
        return device_df

    def fetch_carbon_emission_data(self, st, en):
        # Convert to ISO8601 format and then replace the last characters to fit the required 'Z' format.
        st_iso = datetime.fromisoformat(st).isoformat().replace('+00:00', 'Z')
        en_iso = datetime.fromisoformat(en).isoformat().replace('+00:00', 'Z')

        url = f'https://api.carbonintensity.org.uk/intensity/{st_iso}/{en_iso}/'

         # Make a HTTP request to the URL with retries
        for _ in range(5):
            data = requests.get(url, timeout=10)
            # check if the response was successful 
            if(data.status_code == 200):
                data = data.json()
            else:
                print("HTTP request failed. Response code: " + str(data.status_code))

        # Extract useful data from the response
        timestamp = [datetime.fromisoformat(item['from'].replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M') for item in data['data']]
        co2 = [item['intensity']['actual'] / 1000 / self.tscale for item in data['data']]
        return timestamp, co2
