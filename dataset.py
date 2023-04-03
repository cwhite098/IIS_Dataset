import pandas as pd
import numpy as np
import random

'''
Python file for generatin a dataset to use to make a decision tree using WEKA
'''

def generate_dataset(length):

    items = ['Beans', 'Salad', 'Bread', 'Beer', 'Cheese']
    weights = {'Beans':250, 'Salad':150, 'Bread':400, 'Beer':2100, 'Cheese':200}
    firmnesses = {'Beans':'firm', 'Salad':'soft', 'Bread':'soft', 'Beer':'firm', 'Cheese':'firm'}

    # init df
    df = pd.DataFrame(columns = ['pred_item', 'pred_weight', 'pred_firmness', 'prev_item', 'prev_weight', 'prev_firmness',
                                      'grasp type', 'item_held', 'item_crushed', 'item_dropped', 'items_in_bag', 'human_det', 'estop_pressed', 'label'])

    for i in range(length):
        # First do the human stuff
        # Is human inside workspace?
        human_present = np.random.randint(0,2)
        if human_present:
            p = random.uniform(0,1)
            if p >= 0.95:
                human_detected = 1
            else:
                human_detected = 0
        else:
            human_detected = 0
        # Has estop been pressed?
        p = random.uniform(0,1)
        if p >= 0.8:
            estop = 1
        else:
            estop = 0

        # Is the current item held?
        item_held = np.random.randint(0,2)

        # Get the true item
        current_item = items[np.random.randint(0,len(items))]
        current_weight = weights[current_item]
        current_firmness = firmnesses[current_item]
        # is the item correctly identified?
        p = random.uniform(0,1)
        if p < 0.8:
            supposed_item = current_item
            supposed_weight = current_weight
            supposed_firmness = current_firmness
        else:
            supposed_item = items[np.random.randint(0,len(items))]
            supposed_weight = weights[supposed_item]
            supposed_firmness = firmnesses[supposed_item]

        # Determine the grasp type, if the item is held
        if item_held:
            if supposed_firmness == 'firm':
                grasp_type = 'firm'
            if supposed_firmness == 'soft':
                grasp_type = 'soft'

        # Determine current bag state
        items_in_bag = np.random.randint(0,4)
        previous_item = items[np.random.randint(0,len(items))]
        previous_weight = weights[previous_item]
        previous_firmness = firmnesses[previous_item]

        # What  to do if item is not held
        if not item_held:
            crushed = 0
            dropped = 0
            grasp_type = 'none'
            if supposed_firmness == 'firm':
                label = 'firm_grasp'
            if supposed_firmness == 'soft':
                label = 'soft_grasp'

        # Determine label if item held
        if item_held:
            # Soft grasp
            if grasp_type == 'soft':
                crushed = 0
                if current_firmness == 'soft':
                    label = 'place'
                if current_firmness == 'firm':
                    p = random.uniform(0,1)
                    if p >= 0.5:
                        dropped = 1
                        label = 'help'
                    else:
                        dropped = 0
                        label = 'place'
            # Firm grasp
            if grasp_type == 'firm':
                dropped = 0
                if current_firmness == 'firm':
                    crushed  = 0
                    label = 'place'
                if current_firmness == 'soft':
                    crushed = 1
                    label = 'help'
            
        # Decision based on current bag state
        if items_in_bag == 3:
            label = 'request_bag'
        elif (previous_weight < supposed_weight and previous_firmness == 'soft'):
            label = 'request_bag'
        else:
            pass

        # Decision based on human safety
        if human_detected or estop:
            label = 'estop'

        # Add row to dataset
        df2 = pd.DataFrame([[supposed_item, supposed_weight, supposed_firmness, previous_item, previous_weight, previous_firmness,
                                              grasp_type, item_held, crushed, dropped, items_in_bag, human_detected, estop,  label]],
                                              columns = ['pred_item', 'pred_weight', 'pred_firmness', 'prev_item', 'prev_weight', 'prev_firmness',
                                      'grasp type', 'item_held', 'item_crushed', 'item_dropped', 'items_in_bag', 'human_det', 'estop_pressed', 'label'])
        
        df = df.append(df2)

    return df




def main():

    df = generate_dataset(500)
    df.to_csv('dataset.csv')


if __name__ == '__main__':
    main()