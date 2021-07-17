import datetime, os, threading, random
from flask import Flask, request, redirect, url_for, jsonify, abort, send_from_directory
from werkzeug.wrappers import response

def do_analysis(graphs_path,data_path):
    # CREATING DIRECTORIES

    # main directorey "name = date & time"
    path = graphs_path
    os. mkdir(path)

    # main levels directory-------------------------------
    personal_dir = path + "/personal"
    os. mkdir(personal_dir)

    # subdirectories-------------------------------------------------------------------------------
    #**********************************************************************************************

    # 1- personal subdirectories---------------------------------------------
    gender_dir = personal_dir + "/gender"
    os. mkdir(gender_dir)

    age_dir = personal_dir + "/age"
    os. mkdir(age_dir)

    weight_dir = personal_dir + "/weight"
    os. mkdir(weight_dir)

    height_dir = personal_dir + "/height"
    os. mkdir(height_dir)

    education_dir = personal_dir + "/education"
    os. mkdir(education_dir)

    marital_dir = personal_dir + "/marital status"
    os. mkdir(marital_dir)

    income_dir = personal_dir + "/income"
    os. mkdir(income_dir)

    insurance_dir = personal_dir + "/insurance"
    os. mkdir(insurance_dir)

    genera_health_dir = personal_dir + "/general health"
    os. mkdir(genera_health_dir)

    smoker_dir = personal_dir + "/smoker"
    os. mkdir(smoker_dir)

    days_dir = personal_dir + "/days active"
    os. mkdir(days_dir)

    bmi_dir = personal_dir + "/body mass index"
    os. mkdir(bmi_dir)

    waist_dir = personal_dir + "/waist size"
    os. mkdir(waist_dir)

    drinks_dir = personal_dir + "/drinks per day"
    os. mkdir(drinks_dir)

    #_______________________________________________________________________
    # ignoring some warnings
    import warnings
    warnings.filterwarnings("ignore")
    #________________________________________________________________________
    # importing the required packages
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sb

    df = pd.read_csv(data_path)
    df.head()
    #---------------------------------------------------------------------------------------------------

    #									(((((DATA WRANGLING)))))


    #dropping duplicated rows except for the last row.
    df.drop_duplicates(keep='last', inplace=True, ignore_index=True)
    #----------------------------------------------------------------------------------------------------
    #dropping unwanted features
    df.drop(['name'], axis=True, inplace=True)
    #-----------------------------------------------------------------------------------------------------
    #					(1- replacing null cells in categorical data with unknown and decoding encoded data)
    #gender
    df['gender'][df.gender.isna()] = 'unknown'

    #education
    df['education'][df.education.isna()] = 'unknown'

    #marital
    df['marital'][df.marital.isna()] = 'unknown'

    # income
    df['income'][df.income.isna()] = 'unknown'
    df['income'][df.income == 1] = '$0 to $4,999'
    df['income'][df.income == 2] = '$5k to $9,999'
    df['income'][df.income == 3] = '$10k to $14,999'
    df['income'][df.income == 4] = '$15k to $19,999'
    df['income'][df.income == 5] = '$20k to $24,999'
    df['income'][df.income == 6] = '$25k to $34,999'
    df['income'][df.income == 7] = '$35k to $44,999'
    df['income'][df.income == 8] = '$45k to $54,999'
    df['income'][df.income == 9] = '$55k to $64,999'
    df['income'][df.income == 10] = '$65k to $74,999'
    df['income'][df.income == 14] = '$75k to $99,999'
    df['income'][df.income == 15] = '$100k and Over'

    #insurance
    df['insurance'][df.insurance.isna()] = 'unknown'

    #gen_health
    df['gen_health'][df.gen_health.isna()] = 'unknown'

    #smoker
    df['smoker'][df.smoker.isna()] = 'unknown'

    #days_active
    df['days_active'][df.days_active == 0.0] = '0'
    df['days_active'][df.days_active == 1.0] = '1'
    df['days_active'][df.days_active == 2.0] = '2'
    df['days_active'][df.days_active == 3.0] = '3'
    df['days_active'][df.days_active == 4.0] = '4'
    df['days_active'][df.days_active == 5.0] = '5'
    df['days_active'][df.days_active == 6.0] = '6'
    df['days_active'][df.days_active == 7.0] = '7'
    df['days_active'][df.days_active.isna()] = 'unknown'
    #-------------------------------------------------------------------------------------------------
    #					(2- Changing categorical ordinal data to type categoricalDtype)
    #gender
    Gend_levels = ['female', 'male', 'unknown']
    gend_levels = pd.api.types.CategoricalDtype(ordered=True, categories=Gend_levels)
    df.gender = df.gender.astype(gend_levels)

    #education
    Edu_levels = ['postgraduate education', 'college or equivalent', 'secondary or equivalent',
                'preparatory', 'less than preparatory', 'unknown']
    edu_levels = pd.api.types.CategoricalDtype(ordered=True, categories=Edu_levels)
    df.education = df.education.astype(edu_levels)

    #marital
    Mari_levels = ['married', 'widowed', 'divorced', 'separated', 'never married', 'unknown']
    mari_levels = pd.api.types.CategoricalDtype(ordered=True, categories=Mari_levels)
    df.marital = df.marital.astype(mari_levels)

    #income
    Incm_levels = ['unknown', '$0 to $4,999', '$5k to $9,999', '$10k to $14,999', '$15k to $19,999', 
                '$20k to $24,999','$25k to $34,999', '$35k to $44,999', '$45k to $54,999', 
                '$55k to $64,999', '$65k to $74,999', '$75k to $99,999', '$100k and Over']
    incm_levels = pd.api.types.CategoricalDtype(ordered=True, categories=Incm_levels)
    df.income = df.income.astype(incm_levels)

    #insurance
    Insur_levels = ['yes', 'no', 'unknown']
    insur_levels = pd.api.types.CategoricalDtype(ordered=True, categories=Insur_levels)
    df.insurance = df.insurance.astype(insur_levels)

    #gen_health
    Genh_levels = ['excellent', 'very good', 'good', 'fair', 'poor', 'unknown']
    genh_levels = pd.api.types.CategoricalDtype(ordered=True, categories=Genh_levels)
    df.gen_health = df.gen_health.astype(genh_levels)

    #smoker
    Smok_levels = ['yes', 'no', 'unknown']
    smok_levels = pd.api.types.CategoricalDtype(ordered=True, categories=Smok_levels)
    df.smoker = df.smoker.astype(smok_levels)

    #days_active
    Dact_levels = ['0', '1', '2', '3', '4', '5', '6', '7', 'unknown']
    dact_levels = pd.api.types.CategoricalDtype(ordered=True, categories=Dact_levels)
    df.days_active = df.days_active.astype(dact_levels)
    #-------------------------------------------------------------------------------------------------------
    #							(3- More wrangling: removing outliers (ranges))

    # nulling all bmi cells with value greater than 110 or less than 12
    df['bmi'][df.bmi > 110] = np.nan
    df['bmi'][df.bmi < 12] = np.nan

    # nulling all waist_cm cells with value greater than 180 or less than 30
    df['waist_cm'][df.waist_cm > 180] = np.nan
    df['waist_cm'][df.waist_cm < 30] = np.nan

    # nulling all drinks_day cells with value greater than 100 or less than 0
    df['drinks_day'][df.drinks_day > 100] = np.nan
    df['drinks_day'][df.drinks_day < 0] = np.nan

    # nulling all weight_kg cells with value greater than 210 or less than 40
    df['weight_kg'][df.weight_kg > 250] = np.nan
    df['weight_kg'][df.weight_kg < 0] = np.nan

    # nulling all height_cm cells with value greater than 210 or less than 40
    df['height_cm'][df.height_cm > 210] = np.nan
    df['height_cm'][df.height_cm < 40] = np.nan
    #-------------------------------------------------------------------------------------------------------------

    #												(((PLOTTING PERSONAL DATA ANALYTICS)))

    # 									{{1- single variable analysis}}
    # [a- gender]
    plt.figure(figsize=[16,8])
    sb.set_theme(style="darkgrid")

    #_______________________________first plot____________________________________
    plt.subplot(1,2,1)

    sb.countplot(data=df, x='gender', palette = ['pink', 'cornflowerblue']);
    sorted_counts = df.gender.value_counts()
    locs, labels = plt.xticks()

    # loop through each pair of locations and labels
    for loc, label in zip(locs, labels):

        # get the text property for the label to get the correct count
        count = sorted_counts[label.get_text()]
        pct_string = '{:0.1f}%'.format(100*count/df.shape[0])

        # print the annotation just above the top of the bar
        plt.text(loc, count+(sorted_counts[0]/100), pct_string, ha = 'center', color = 'black')

    plt.title('Gender Proportions & Counts', fontsize= 15, pad=10)

    #__________________________________second plot________________________________________
    plt.subplot(1,2,2)

    colors = ['pink', 'cornflowerblue']

    plt.pie(x=sorted_counts, labels=sorted_counts.index, startangle=90, counterclock=True, colors=colors,
            autopct=lambda p: '{:.1f}%'.format(p), wedgeprops = { 'linewidth' : 2, 'edgecolor' : 'white' })
    plt.axis('square')
    plt.xlabel('gender', labelpad=30)
    plt.title('Gender Proportions', fontsize= 15, pad= 25);

    plt.savefig(gender_dir+"/" + "genders");
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    # [b- age]

    plt.figure(figsize=[12, 7])
    sb.set_theme(style="darkgrid")

    #_______________________________first plot____________________________________
    bins = np.arange(0, df.age.max()+3, 3)
    ticks = np.arange(0, df.age.max()+3, 3)
    labels = ['{:.0f}'.format(v) for v in ticks]
    plt.hist(data=df, x='age', bins= bins)
    plt.xticks(ticks, labels)

    plt.axvline(x=df.age.mean(), linestyle='--', linewidth=2, color='r')

    plt.title('Age Distribution', fontsize= 15)
    plt.ylabel('Count', labelpad=10)
    plt.xlabel('Age', labelpad=10);
    plt.savefig(age_dir+"/" + "Age Distribution-histogram");
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    plt.figure(figsize=[20, 8])
    sb.set_theme(style="whitegrid")

    #_______________________________first plot____________________________________
    plt.subplot(2,1,1)
    sb.boxplot(data=df, x='age',  color=sb.color_palette('rainbow', 10)[2])

    ticks = np.arange(0,120,10)
    labels = ['{}'.format(v) for v in ticks]
    plt.xticks(ticks, labels)
    plt.title('Age Distribution', fontsize= 15)
    plt.xlabel('Age', fontsize=12, labelpad=10);


    #_______________________________second plot____________________________________
    plt.subplot(2,1,2)
    sb.violinplot(data=df, x='age', orient='horizontal', color=sb.color_palette('rainbow', 10)[2], inner= None)

    plt.xticks(ticks, labels)
    plt.title('', fontsize= 15)
    plt.xlabel('Age', fontsize=12, labelpad=10);
    plt.ylabel('', fontsize=12, labelpad=10);
    plt.savefig(age_dir+"/" + "Age Distribution-box-violin");
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    plt.figure(figsize=[20,9])
    sb.set_theme(style="whitegrid")

    #_______________________________first plot____________________________________
    plt.subplot(1,2,1)
    sb.kdeplot(data=df, x='age', cut=0, fill= True, color="#00AFBB");
    plt.title('Age Distribution Density', fontsize= 15)
    plt.ylabel('Density', labelpad=10)
    plt.xlabel('Age', labelpad=10);
    plt.axvline(x=df.age.mean(), linestyle='--', linewidth=2, color='r')

    #_______________________________second plot____________________________________
    plt.subplot(1,2,2)
    sb.kdeplot(data=df, x='age', cumulative=True);
    plt.title('Age Cumulative Distribution Function', fontsize= 15)
    plt.ylabel('Propability', labelpad=10)
    plt.xlabel('Age', labelpad=10);
    plt.savefig(age_dir+"/" + "Age Cumulative Distribution Function");
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    age_max_log = np.log10(df.age.max())
    age_min_log = np.log10(df.age.min())
    step = (age_max_log - age_min_log) / 40

    plt.figure(figsize=[16, 8])
    sb.set_theme(style="whitegrid")

    bins= 10 ** np.arange(age_min_log, age_max_log + step, step)
    #ticks = 10 ** np.arange(age_min_log, age_max_log + step, step * 4)
    ticks = [1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 110]
    labels = ['{:.0f}'.format(v) for v in ticks]
    plt.hist(data=df, x='age', bins=bins)
    plt.xscale('log')
    plt.xticks(ticks, labels)

    plt.title('Age Distribution (Log_Transformed)', fontsize= 15)
    plt.ylabel('Count', labelpad=10)
    plt.xlabel('Age "Log-transformed"', labelpad=10);
    plt.savefig(age_dir+"/" + "Age Distribution (Log_Transformed)");
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    plt.figure(figsize=[20,8])
    sb.set_theme(style="whitegrid")

    #_______________________________first plot____________________________________
    plt.subplot(1,2,1)
    sb.kdeplot(data=df, x='age', cut=0, fill= True, log_scale=True);
    ticks = [1, 3, 10, 30, 60,  80, 110]
    labels = ['{:.0f}'.format(v) for v in ticks]
    plt.xticks(ticks, labels)
    plt.title('Age Distribution Density (Log_Transformed)', fontsize= 15)
    plt.ylabel('Density', labelpad=10)
    plt.xlabel('Age "Log-transformed"', labelpad=10);


    #_______________________________second plot____________________________________
    plt.subplot(1,2,2)
    sb.kdeplot(data=df, x='age', cumulative=True, log_scale=True);
    plt.title('Age CDF (Log_Transformed)', fontsize= 15)
    plt.ylabel('Propability', labelpad=10)
    plt.xlabel('Age "Log-transformed"', labelpad=10);
    plt.savefig(age_dir+"/" + "Age Distribution Density (Log_Transformed)");
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    # [c- Education]
    plt.figure(figsize=[15, 8])
    sb.set_theme(style="whitegrid")

    sb.countplot(data=df, x='education', palette='Blues_r')

    education_counts = df.education.value_counts()
    locs, labels = plt.xticks()

    # loop through each pair of locations and labels
    for loc, label in zip(locs, labels):

        # get the text property for the label to get the correct count
        count = education_counts[label.get_text()]
        pct_string = '{:0.1f}%'.format(100*count/df.shape[0])

        # print the annotation just above the top of the bar
        plt.text(loc, count+(education_counts[0]/100), pct_string, ha = 'center', color = 'black')

    plt.xticks(rotation=20)
    plt.title("Patients' Education Distribution", fontsize= 15)
    plt.ylabel('Count', labelpad=10)
    plt.xlabel('Education Level', labelpad=20);
    plt.savefig(education_dir+"/" + "Patients' Education Distribution");
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    #	{d- Marital}

    plt.figure(figsize=[16, 8])
    sb.set_theme(style="whitegrid")

    sb.countplot(data=df, x='marital', palette='BuGn_r')

    marital_counts = df.marital.value_counts()
    locs, labels = plt.xticks()

    # loop through each pair of locations and labels
    for loc, label in zip(locs, labels):

        # get the text property for the label to get the correct count
        count = marital_counts[label.get_text()]
        pct_string = '{:0.1f}%'.format(100*count/df.shape[0])

        # print the annotation just above the top of the bar
        plt.text(loc, count+(marital_counts[0]/100), pct_string, ha = 'center', color = 'black')
        
    plt.xticks(rotation=20);
    plt.title("Patients' Marital Status Distribution", fontsize= 15)
    plt.ylabel('Count', labelpad=10)
    plt.xlabel('Marital Status', labelpad=20);
    plt.savefig(marital_dir+"/" + "Patients' Marital Status Distribution");
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    #	{e- Income}

    plt.figure(figsize=[20, 10])
    sb.set_theme(style="whitegrid")

    sb.countplot(data=df, x='income', palette='BuPu')

    income_counts = df.income.value_counts()
    locs, labels = plt.xticks()

    # loop through each pair of locations and labels
    for loc, label in zip(locs, labels):

        # get the text property for the label to get the correct count
        count = income_counts[label.get_text()]
        pct_string = '{:0.1f}%'.format(100*count/df.shape[0])

        # print the annotation just above the top of the bar
        plt.text(loc, count+(income_counts[0]/100), pct_string, ha = 'center', color = 'black')
        
    plt.xticks(rotation=30);
    plt.title("Patients' Family Income Distribution", fontsize= 15)
    plt.ylabel('Count', labelpad=10)
    plt.xlabel('Income Range', labelpad=20);
    plt.savefig(income_dir+"/" + "Patients' Family Income Distribution");
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    #	{f- Insurance}

    sorted_counts = df.insurance.value_counts()

    #to manage plot size
    plt.figure(figsize=[16,8])
    sb.set_theme(style="darkgrid")

    #_______________________________first plot____________________________________
    plt.subplot(1,2,1)

    sb.countplot(data=df, x='insurance', order=sorted_counts.index,
                palette = ['limegreen', 'tomato', 'black']);
    insurance_counts = df.insurance.value_counts()
    locs, labels = plt.xticks()

    # loop through each pair of locations and labels
    for loc, label in zip(locs, labels):

        # get the text property for the label to get the correct count
        count = insurance_counts[label.get_text()]
        pct_string = '{:0.1f}%'.format(100*count/df.shape[0])

        # print the annotation just below the top of the bar
        plt.text(loc, count+(sorted_counts[0]/100), pct_string, ha = 'center', color = 'black')

    plt.xlabel('Insured?', labelpad=15)
    plt.ylabel('Count', labelpad=10)
    plt.title("Patients' Insurance Distribution", fontsize= 15, pad=15)


    #__________________________________second plot________________________________________
    plt.subplot(1,2,2)

    colors = ['limegreen', 'tomato', 'black']

    plt.pie(x=sorted_counts, labels=sorted_counts.index, startangle=90, counterclock=True, 
            colors=colors, autopct=lambda p: '{:.1f}%'.format(p))
    plt.axis('square')
    plt.xlabel('Insured?', labelpad=30)
    plt.title("Insurance Proportions", fontsize= 15, pad=50);
    plt.savefig(insurance_dir+"/" + "Patients' Insurance Distribution");
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    #	{g- General Health}

    plt.figure(figsize=[15, 8])
    sb.set_theme(style="whitegrid")

    sb.countplot(data=df, x='gen_health', palette='Greens_r')

    health_counts = df.gen_health.value_counts()
    locs, labels = plt.xticks()

    # loop through each pair of locations and labels
    for loc, label in zip(locs, labels):

        # get the text property for the label to get the correct count
        count = health_counts[label.get_text()]
        pct_string = '{:0.1f}%'.format(100*count/df.shape[0])

        # print the annotation just above the top of the bar
        plt.text(loc, count+(health_counts[0]/100), pct_string, ha = 'center', color = 'black')
        
    plt.xticks(rotation=0);
    plt.title("Patients' Health State Distribution", fontsize= 15)
    plt.ylabel('Count', labelpad=10)
    plt.xlabel('Health State', labelpad=20);
    plt.savefig(genera_health_dir+"/" + "Patients' Health State Distribution");
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    #	{h- Smoker}

    plt.figure(figsize=[15,10])
    sb.set_theme(style="darkgrid")

    #_______________________________first plot____________________________________
    plt.subplot(1,2,1)

    sb.countplot(data=df, x='smoker', palette = ['lightslategray', 'lawngreen', 'cornflowerblue']);
    smok_counts = df.smoker.value_counts()
    locs, labels = plt.xticks()

    # loop through each pair of locations and labels
    for loc, label in zip(locs, labels):

        # get the text property for the label to get the correct count
        count = smok_counts[label.get_text()]
        pct_string = '{:.0f}'.format(count)

        # print the annotation just above the top of the bar
        plt.text(loc, count+(smok_counts[0]/100), pct_string, ha = 'center', color = 'black')

    plt.title('Smoking State Distribution', fontsize= 15)
    plt.xlabel('Smoker?', labelpad=20)

    #__________________________________second plot________________________________________
    plt.subplot(1,2,2)

    colors = ['lawngreen', 'cornflowerblue', 'lightslategray']

    plt.pie(x=smok_counts, labels=smok_counts.index, startangle=90, counterclock=True, colors=colors,
            autopct=lambda p: '{:.1f}%'.format(p), 
            wedgeprops = { 'linewidth' : 2, 'edgecolor' : 'white' })

    plt.axis('square')
    plt.xlabel('Smoker?', labelpad=30)
    plt.title('Smoking State Proportions', fontsize= 15);
    plt.savefig(smoker_dir+"/" + "Patients' Health State Distribution");
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    #	{i- days_active}
    plt.figure(figsize=[15, 8])
    sb.set_theme(style="whitegrid")

    sb.countplot(data=df, x='days_active', palette='Greens')

    days_counts = df.days_active.value_counts()
    locs, labels = plt.xticks()

    # loop through each pair of locations and labels
    for loc, label in zip(locs, labels):

        # get the text property for the label to get the correct count
        count = days_counts[label.get_text()]
        pct_string = '{:0.1f}%'.format(100*count/df.shape[0])

        # print the annotation just above the top of the bar
        plt.text(loc, count+(days_counts[0]/100), pct_string, ha = 'center', color = 'black')
        
    plt.xticks(rotation=0);
    plt.title("Distribution Of Patients By Number of Active Days ", fontsize= 15, pad=10)
    plt.ylabel('Count', labelpad=10)
    plt.xlabel('Number of Days per Week', labelpad=20);
    plt.savefig(days_dir+"/" + "Distribution Of Patients By Number of Active Days");
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    #	{j- bmi}
    plt.figure(figsize=[15, 8])
    sb.set_theme(style="darkgrid")

    #_______________________________first plot____________________________________
    bins = np.arange(0, df.bmi.max()+.5, .5)
    plt.hist(data=df, x='bmi', bins= bins)
    ticks = np.arange(0, df.bmi.max()+5, 5)
    labels = ['{:.0f}'.format(v) for v in ticks]
    plt.xticks(ticks, labels)
    plt.xlim(5,)
    plt.axvline(x=df.bmi.mean(), linestyle='-', linewidth=3, color='yellow')

    plt.title('Body-Mass-Index Distribution', fontsize= 15)
    plt.ylabel('Count', labelpad=10)
    plt.xlabel('BMI', labelpad=15);
    plt.savefig(bmi_dir+"/" + "Body-Mass-Index Distribution");
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    #	{waist_cm}
    plt.figure(figsize=[16, 8])
    sb.set_theme(style="darkgrid")

    #_______________________________first plot____________________________________
    bins = np.arange(0, df.waist_cm.max()+1, 1)
    ticks = np.arange(0, df.waist_cm.max()+10, 10)
    labels = ['{:.0f}'.format(v) for v in ticks]
    plt.hist(data=df, x='waist_cm', bins= bins)
    plt.xticks(ticks, labels)
    plt.xlim(30,)

    plt.title('waist circumference Distribution', fontsize= 15)
    plt.ylabel('Count', labelpad=10)
    plt.xlabel('Waist (cm)', labelpad=15);
    plt.savefig(waist_dir+"/" + "waist circumference Distribution");
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    #	{drinks_day}
    plt.figure(figsize=[20, 10])
    sb.set_theme(style="darkgrid")

    index_ordered = df.drinks_day.value_counts().index.sort_values()
    index_ordered = index_ordered.astype('int32')

    sb.countplot(data=df, y='drinks_day', order=index_ordered, color=sb.color_palette()[0])
    plt.title('Distribution Of Number Of Drinks Per Day', fontsize= 15)
    plt.ylabel('Drinks / Day', labelpad=20)
    plt.xlabel('Count', labelpad=15);
    plt.savefig(drinks_dir+"/" + "Distribution Of Number Of Drinks Per Day");
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    #	{weight_kg}
    plt.figure(figsize=[20, 10])
    sb.set_theme(style="darkgrid")

    bins = np.arange(0, df.weight_kg.max()+2, 2)
    ticks = np.arange(0, df.weight_kg.max()+10, 10)
    labels = ['{:.0f}'.format(v) for v in ticks]
    plt.hist(data=df, x='weight_kg', bins= bins)
    plt.xticks(ticks, labels)

    plt.title("Patients' Weights Distribution", fontsize= 15)
    plt.ylabel('Count', labelpad=10)
    plt.xlabel('Weight (kg)', labelpad=15);
    plt.savefig(weight_dir+"/" + "Patients' Weights Distribution");
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    #	{height_cm}
    plt.figure(figsize=[20, 10])
    sb.set_theme(style="darkgrid")

    #_______________________________first plot____________________________________
    bins = np.arange(0, df.height_cm.max()+2, 2)
    ticks = np.arange(40, df.height_cm.max()+10, 10)
    labels = ['{:.0f}'.format(v) for v in ticks]
    plt.hist(data=df, x='height_cm', bins= bins)
    plt.xticks(ticks, labels)
    plt.xlim(40,)

    plt.title("Patients' Heights Distribution", fontsize= 15)
    plt.ylabel('Count', labelpad=10)
    plt.xlabel('Height (cm)', labelpad=15);
    plt.savefig(height_dir+"/" + "Patients' Heights Distribution");
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    #		{Bivariate exploration}
    #	{gender with age}
    plt.figure(figsize=[20, 10])
    sb.set_theme(style="whitegrid")

    #_______________________________first plot____________________________________
    bins = np.arange(0, df.age.max()+1, 1)
    ticks = np.arange(0, df.age.max()+5, 5)
    labels = ['{:.0f}'.format(v) for v in ticks]
    plt.hist(data=df[df.gender == 'female'], x='age', bins= bins, color='pink')
    #_______________________________first plot____________________________________

    plt.hist(data=df[df.gender == 'male'], x='age', bins= bins, color='cornflowerblue', alpha=.5)
    plt.legend(['female','male']);

    plt.xticks(ticks, labels)

    plt.title('Age Distribution For Both Genders', fontsize= 15)
    plt.ylabel('Count', labelpad=10)
    plt.xlabel('Age', labelpad=10);
    plt.savefig(gender_dir+"/" + "Age Distribution For Both Genders-his");
    plt.savefig(age_dir+"/" + "Age Distribution For Both Genders-hist");
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    sb.set_theme(style="whitegrid")

    df['age'][df.gender == 'female'].value_counts().sort_index().plot(kind='line', figsize=(20,10), 
                                                                        linewidth = '5', fontsize=20, 
                                                                        color='pink');
    df['age'][df.gender == 'male'].value_counts().sort_index().plot(kind='line', figsize=(20,10),
                                                                    linewidth = '5', fontsize=20);

    ticks = np.arange(0, 110, 10)
    labels = ['{:.0f}'.format(v) for v in ticks]
    plt.xticks(ticks, labels)
    plt.title('Age Distribution For Females and Males', fontsize= 25)
    plt.xlabel('Age', fontsize= 20)
    plt.ylabel('Count', fontsize= 20)
    plt.legend(['Female','Male'],fontsize=20);
    plt.savefig(age_dir+"/" + "Age Distribution For Both Genders-line");
    plt.savefig(gender_dir+"/" + "Age Distribution For Both Genders-line");
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    plt.figure(figsize=[15, 10])
    sb.set_theme(style="whitegrid")
    
    #_______________________________first plot____________________________________
    plt.subplot(1,5,1)
    sb.boxplot(data=df, x='gender', y='age', order=['female', 'male'], palette=['pink', 'b'], boxprops=dict(alpha=.99))
    plt.title('Age For\nBoth Genders', fontsize= 15, pad=15)
    plt.ylabel('Age', fontsize=12, labelpad=10);
    plt.xlabel('Gender', fontsize=12, labelpad=20)

    #_______________________________second plot____________________________________
    plt.subplot(1,5,3)
    sb.violinplot(data=df, x='gender', y='age', inner=None, order=['female', 'male'], palette=['pink', 'b'])
    plt.title('Age For\nBoth Genders', fontsize= 15, pad=15)
    plt.ylabel('Age', fontsize=12, labelpad=10);
    plt.xlabel('Gender', fontsize=12, labelpad=20)

    #_______________________________second plot____________________________________
    plt.subplot(1,5,5)
    sb.barplot(data=df, x='gender', y='age', errwidth=0, order=['female', 'male'], palette=['pink', 'b'])

    plt.title('Average Age For\nBoth Genders', fontsize= 15, pad=15)
    plt.xlabel('Gender', fontsize=12, labelpad=20)
    plt.ylabel('Average Age', fontsize=12, labelpad=10);
    plt.savefig(age_dir+"/" + "Age Distribution For Both Genders-box-violin-bar");
    plt.savefig(gender_dir+"/" + "Age Distribution For Both Genders-line-box-violin-bar");
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    plt.figure(figsize=[20,10])
    sb.set_theme(style="whitegrid")

    #_______________________________first plot____________________________________
    plt.subplot(1,2,1)
    sb.kdeplot(data=df, x='age', hue='gender', hue_order=['male', 'female'], cut=0, fill= True);
    plt.title('Age Distribution Density', fontsize= 15)
    plt.ylabel('Density', labelpad=10)
    plt.xlabel('Age', labelpad=10);

    #_______________________________second plot____________________________________
    plt.subplot(1,2,2)
    sb.kdeplot(data=df, hue='gender', hue_order=['male', 'female'], x='age', cumulative=True);
    plt.title('Age Cumulative Distribution Function', fontsize= 15)
    plt.ylabel('Propability', labelpad=10)
    plt.xlabel('Age', labelpad=10);
    plt.savefig(age_dir+"/" + "Age Distribution For Both Genders-density");
    plt.savefig(gender_dir+"/" + "Age Distribution For Both Genders-density");
    
    return path

def get_files(dir):
    files=[]
    for path in os.listdir(dir):
        full_path = os.path.join(dir, path)
        if os.path.isfile(full_path):
            files.append(full_path)
        else:
            files+= get_files(full_path)
    return files

app=Flask(__name__)
graphs_dict={}
thread_dict={}
graphs_root_path= os.path.abspath(os.getenv('GRAPHS_PATH',os.getcwd()))

@app.route('/')
def index():
    return redirect(url_for('running'))

@app.route('/status')
def running():
    return 'app is running'

@app.route('/file')
def get_file():
    file_path= request.args.get('path')
    if file_path is None:
        abort(404)
    return send_from_directory(graphs_root_path,file_path)

@app.route('/analysis/<int:task_id>')
def get_analysis(task_id):
    if task_id not in graphs_dict:
        abort(404)
    return jsonify({
        "running": thread_dict[task_id].is_alive(),
        "graphs": [filename.replace(graphs_root_path+'/','') for filename in get_files(graphs_dict[task_id])]
    })

@app.route('/analysis/<int:task_id>/html')
def get_analysis_html(task_id):
    if task_id not in graphs_dict:
        abort(404)
    response_str='<h1>Result Graphs</h1>\n'
    for graph in [filename.replace(graphs_root_path+'/','') for filename in get_files(graphs_dict[task_id])]:
        # graph_path=os.path.join(graphs_root_path,graph)
        response_str+= '<h3>{}</h3>\n'.format(os.path.basename(graph))
        response_str+= '<img src="{}" alt="{}"><br>\n'.format('/file?path='+graph, os.path.basename(graph))
    if thread_dict[task_id].is_alive():
        response_str+='<h1>Task is still running</h1>'
    else:
        response_str+='<h1>Task is Finshed</h1>'
    return response_str

@app.route('/analysis',methods=["GET","POST"])
def post_analysis():
    data= request.get_json()
    now = datetime.datetime.now()
    directory = now.strftime("%d-%m-%Y-(%H-%M-%S)")
    graphs_path=os.path.join(graphs_root_path, directory)
    if data is not None:
        # @TODO
        pass
    else:
        data_path='Patients_Dataset.csv'
    thread= threading.Thread(target=do_analysis ,args=(graphs_path,data_path,))
    thread.start()
    thread.is_alive()
    task_id= random.randint(1,99999999)
    thread_dict[task_id]= thread
    graphs_dict[task_id]= graphs_path
    return jsonify({
        'message': 'task started successfully',
        'task_id':task_id
    })
        
if __name__ == "__main__":
    app.run(port=8000, debug=True, host='0.0.0.0')