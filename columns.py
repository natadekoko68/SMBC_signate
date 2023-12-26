cols = {'boro_ct': False,
        'borocode': False,
        'boroname': True,
        'cb_num': False,
        'cncldist': False,
        'created_date': False,
        'created_year': True,
        'curb_loc': False,
        'guards': True,
        'health': True,
        'nta': False,
        'problems_Branch': True,
        'problems_BranchOther': True,
        'problems_Grates': True,
        'problems_Lights': True,
        'problems_Metal': True,
        'problems_RootOther': True,
        'problems_Rope': False,
        'problems_Sneakers': True,
        'problems_Stones': False,
        'problems_Trunk': True,
        'problems_TrunkOther': False,
        'problems_Wires': True,
        'problems_num': True,
        'season': False,
        'sidewalk': True,
        'spc_common': True,
        'spc_latin': False,
        'st_assem': True,
        'st_senate': True,
        'staff': True,
        'steward': True,
        'tree_dbh': False,
        'user_type': False,
        'zip_city': False}

cols2 = {'boro_ct': False,
         'borocode': False,
         'boroname': True,
         'cb_num': True,
         'cncldist': False,
         'created_date': False,
         'created_year': False,
         'curb_loc': False,
         'guards': False,
         'health': True,
         'nta': False,
         'problems_Branch': True,
         'problems_BranchOther': False,
         'problems_Grates': True,
         'problems_Lights': True,
         'problems_Metal': True,
         'problems_RootOther': True,
         'problems_Rope': False,
         'problems_Sneakers': True,
         'problems_Stones': True,
         'problems_Trunk': True,
         'problems_TrunkOther': False,
         'problems_Wires': False,
         'problems_num': False,
         'season': False,
         'sidewalk': False,
         'spc_common': False,
         'spc_latin': True,
         'st_assem': False,
         'st_senate': True,
         'staff': False,
         'steward': True,
         'tree_dbh': False,
         'user_type': True,
         'zip_city': True
         }

cols3 = {'boro_ct': True,
 'borocode': False,
 'boroname': False,
 'cb_num': False,
 'cncldist': True,
 'created_date': False,
 'created_year': True,
 'curb_loc': True,
 'guards': True,
 'health': True,
 'nta': True,
 'problems_Branch': True,
 'problems_BranchOther': False,
 'problems_Grates': False,
 'problems_Lights': True,
 'problems_Metal': False,
 'problems_RootOther': False,
 'problems_Rope': False,
 'problems_Sneakers': True,
 'problems_Stones': True,
 'problems_Trunk': False,
 'problems_TrunkOther': True,
 'problems_Wires': True,
 'problems_num': True,
 'season': True,
 'sidewalk': False,
 'spc_common': False,
 'spc_latin': True,
 'st_assem': True,
 'st_senate': False,
 'staff': False,
 'steward': False,
 'tree_dbh': False,
 'user_type': False,
 'zip_city': True}

cols4 = {'boro_ct': True,
 'borocode': True,
 'boroname': True,
 'cb_num': True,
 'cncldist': True,
 'created_date': False,
 'created_year': False,
 'curb_loc': True,
 'guards': False,
 'health': True,
 'nta': False,
 'problems_Branch': True,
 'problems_BranchOther': True,
 'problems_Grates': True,
 'problems_Lights': False,
 'problems_Metal': True,
 'problems_RootOther': False,
 'problems_Rope': True,
 'problems_Sneakers': True,
 'problems_Stones': False,
 'problems_Trunk': False,
 'problems_TrunkOther': True,
 'problems_Wires': True,
 'problems_num': True,
 'season': True,
 'sidewalk': True,
 'spc_common': True,
 'spc_latin': True,
 'st_assem': False,
 'st_senate': False,
 'staff': True,
 'steward': True,
 'tree_dbh': False,
 'user_type': False,
 'zip_city': True}

cols5 = {'boro_ct': False,
 'borocode': False,
 'boroname': False,
 'cb_num': True,
 'cncldist': True,
 'created_date': False,
 'created_year': False,
 'curb_loc': True,
 'guards': False,
 'health': True,
 'nta': False,
 'problems_Branch': True,
 'problems_BranchOther': False,
 'problems_Grates': True,
 'problems_Lights': True,
 'problems_Metal': False,
 'problems_RootOther': False,
 'problems_Rope': True,
 'problems_Sneakers': True,
 'problems_Stones': False,
 'problems_Trunk': False,
 'problems_TrunkOther': False,
 'problems_Wires': False,
 'problems_num': True,
 'season': False,
 'sidewalk': False,
 'spc_common': True,
 'spc_latin': True,
 'st_assem': False,
 'st_senate': False,
 'staff': False,
 'steward': True,
 'tree_dbh': True,
 'user_type': True,
 'zip_city': True}

cols6 = {'boro_ct': True,
 'borocode': True,
 'boroname': True,
 'cb_num': False,
 'cncldist': True,
 'created_date': False,
 'created_year': True,
 'curb_loc': False,
 'guards': False,
 'health': True,
 'nta': False,
 'problems_Branch': True,
 'problems_Grates': False,
 'problems_Lights': True,
 'problems_Metal': True,
 'problems_Rope': True,
 'problems_Sneakers': True,
 'problems_Stones': False,
 'problems_Trunk': False,
 'problems_Wires': True,
 'problems_num': False,
 'season': True,
 'sidewalk': True,
 'spc_common': True,
 'spc_latin': False,
 'st_assem': True,
 'st_senate': True,
 'staff': False,
 'steward': True,
 'tree_dbh': False,
 'user_type': False,
 'zip_city': True}

def bn(x):
    return 1 if x else 0

def bn_num():
    sum = 0
    for i in [cols,cols2,cols3,cols4,cols5]:
        sum += bn(i[key])
    return sum


for key in cols:
    if bn_num() >= 3:
        print(key)