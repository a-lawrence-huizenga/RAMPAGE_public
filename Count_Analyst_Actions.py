import pandas as pd


# function to get unique values
def unique(list1):
    # intilize a null list
    unique_list = []

    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
            # print list
    return unique_list


analyst_df = pd.read_csv('C:/Users/ashley/desktop/AnalystC-logs_step.csv')
steps = unique(list(analyst_df['Step']))
allActions = unique(list(analyst_df['Action']))
analysts = ['A', 'B', 'C', 'D', 'E', 'F']

# Setting up file for exporting data
columnHeadings = ['Analyst', 'Step', *allActions]
filename = 'Cycle2_Action_Counts.csv'
# creating a dataframe with heading names
df_output = pd.DataFrame([columnHeadings])
# exporting the data frame to a csv file
df_output.to_csv(filename, index=False, header=False, mode='a')  # check name

for analyst in analysts:
    analyst_df = pd.read_csv('C:/Users/ashley/desktop/Analyst' + str(analyst) + '-logs_step.csv')
    for step in steps:
        oneStep = analyst_df[analyst_df['Step'] == step]
        numActionsStep = [analyst, step]
        for action in allActions:
            if action == 'Edited Hypothesis':
                oneStepEdited = oneStep[oneStep['Action'] == 'Edited Hypothesis']
                oneStepCreated = oneStep[oneStep['Action'] == 'Created Hypothesis']
                editedHypotheses = unique(list(oneStepEdited['Object1']))
                createdHypotheses = unique(list(oneStepCreated['Object1']))
                uniqueHypotheses = list(set(editedHypotheses) - set(createdHypotheses))
                numActionsStep.append(len(uniqueHypotheses))
            elif action == 'Edited Forecast':
                oneStepEdited = oneStep[oneStep['Action'] == 'Edited Forecast']
                oneStepCreated = oneStep[oneStep['Action'] == 'Created Forecast']
                editedForecast = unique(list(oneStepEdited['Object1']))
                createdForecast = unique(list(oneStepCreated['Object1']))
                uniqueForecast = list(set(editedForecast) - set(createdForecast))
                numActionsStep.append(len(uniqueForecast))
            else:
                numActionsStep.append(len(oneStep[oneStep['Action'] == action]))
        df_output = pd.DataFrame([numActionsStep])
        df_output.to_csv(filename, index=False, header=False, mode='a')