import numpy as np
import cerebral as cb 

def ensure_default_values(data):
    for i, row in data.iterrows():
        
        try:
            _ = data.at[i, 'Dmax']
            hasDmax = True
        except BaseException:
            hasDmax = False

        if(hasDmax):
            if not np.isnan(data.at[i, 'Dmax']):
                if row['Dmax'] == 0:
                    data.at[i, 'GFA'] = 0
                elif row['Dmax'] <= 0.15:
                    data.at[i, 'GFA'] = 1
                else:
                    data.at[i, 'GFA'] = 2
            else:
                data.at[i, 'Dmax'] = cb.features.maskValue
        else:
            data.at[i, 'Dmax'] = cb.features.maskValue

        try:
            _ = data.at[i, 'GFA']
            hasGFA = True
        except BaseException:
            hasGFA = False

        if(hasGFA):
            if not np.isnan(data.at[i, 'GFA']):
                if(int(data.at[i, 'GFA']) == 0):
                    data.at[i, 'Dmax'] = 0
                elif(int(data.at[i, 'GFA']) == 1):
                    data.at[i, 'Dmax'] = 0.15
                elif(int(data.at[i, 'GFA']) == 2):
                    if('Dmax' in row):
                        if(np.isnan(data.at[i, 'Dmax']) or data.at[i, 'Dmax'] == 0 or data.at[i, 'Dmax'] is None):
                            data.at[i, 'Dmax'] = cb.features.maskValue
                    else:
                        data.at[i, 'Dmax'] = cb.features.maskValue
                else:
                    data.at[i, 'Dmax'] = cb.features.maskValue
            else:
                data.at[i, 'GFA'] = cb.features.maskValue
        else:
            data.at[i, 'GFA'] = cb.features.maskValue

        if 'Tx' in row and 'Tg' in row:
            if (not np.isnan(row['Tx'])
                and not np.isnan(row['Tg']) 
                and not row['Tx'] == cb.features.maskValue
                and not row['Tg'] == cb.features.maskValue):
                data.at[i, 'deltaT'] = row['Tx'] - row['Tg']
            else:
                data.at[i, 'deltaT'] = cb.features.maskValue
                
    return data
                
