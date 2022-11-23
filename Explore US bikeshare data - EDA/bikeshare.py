import time
import pandas as pd
import numpy as np

CITY_DATA = { 'chi': 'chicago.csv',
              'nyc': 'new_york_city.csv',
              'was': 'washington.csv' }

def get_filters():
    """
    Asks user to specify a city, month, and day to analyze.

    Returns:
        (str) city - name of the city to analyze
        (str) month - name of the month to filter by, or "all" to apply no month filter
        (str) day - name of the day of week to filter by, or "all" to apply no day filter
    """
    print('Let\'s look into some US bikeshare stats!')
    
    valid_city_options=['chi','nyc','was']
    valid_month_options=['all','jan','feb','mar','apr','may','june']
    valid_day_options=['all','mon','tue','wed','thu','fri','sat','sun']
    
    # TO DO: get user input for city (chicago, new york city, washington).
    city = input("City abbreviation? chi,nyc,was:\n")  
    while (city.lower() not in valid_city_options):
        print('Unexpected Input..')#ValueError combating
        city=input("City abbreviation? chi,nyc,was:\n")
    
    # TO DO: get user input for month (all, january, february, ... , june)
    month = input("all or specific Month? jan,feb,mar,apr,may,june:\n")
    while (month.lower() not in valid_month_options):
        print('Unexpected Input..')#ValueError combating
        month=input("all or specific Month? jan,feb,mar,apr,may,june:\n")

    # TO DO: get user input for day of week (all, monday, tuesday, ... sunday)
    day = input("all or specific Day? sun,mon,tues,wed,thu,fri,sat:\n")
    while (day.lower() not in valid_day_options):
        print('Unexpected Input..')#ValueError combating
        day=input("all or specific Day? sun,mon,tues,wed,thu,fri,sat:\n")


    print('-'*40)
    return city,month,day


def load_data(city,month,day):
    """
    Loads data for the specified city and filters by month and day if applicable.

    Args:
        (str) city - name of the city to analyze
        (str) month - name of the month to filter by, or "all" to apply no month filter
        (str) day - name of the day of week to filter by, or "all" to apply no day filter
    Returns:
        df - Pandas DataFrame containing city data filtered by month and day
    """
    # load data file into a dataframe
    df = pd.read_csv(CITY_DATA[city])

    # convert the Start Time column to datetime
    df['Start Time'] = pd.to_datetime(df['Start Time'])

    # extract month and day of week from Start Time to create new columns
    df['month'] = df['Start Time'].dt.month
    df['day_of_week'] = df['Start Time'].dt.dayofweek

    # filter by month if applicable
    if month!='all':
        # use the index of the months list to get the corresponding int
        
        month_num=0
        for i,xmonth in enumerate(months_list):
            if xmonth==month:
                month_num=i+1
            else:
                continue
        
        df=df.loc[df['month'] == month_num]
    
    # filter by day of week if applicable
    if day!='all':
        
        day_num=[i for i, e in enumerate(days_list) if e == day]
        #print('day is {}, daynum is {}'.format(day,day_num))
        df=df.loc[df['day_of_week']==day_num]    
    
    #replace blanks by NaN
    df = df.replace(r'^\s*$', np.nan, regex=True)

    return df


def time_stats(df):
    """Displays statistics on the most frequent times of travel."""

    print('\nCalculating The Most Frequent Times of Travel...\n')
    start_time = time.time()
    
    #if (len(df['month'].dropna().unique())>1):
    # TO DO: display the most common month
    most_common_month=df['month'].value_counts().idxmax()
    print('The most common month is: {}'.format(months_list[most_common_month-1])) 
    
    #if (len(df['day_of_week'].dropna().unique())>1):
    # TO DO: display the most common day of week
    most_common_dayofweek=df['day_of_week'].value_counts().idxmax()
    print('The most common day of week is: {}'.format(days_list[most_common_dayofweek]))


    # TO DO: display the most common start hour
    datetime=pd.to_datetime(df['Start Time'])
    df['hour'] =datetime.dt.hour
    most_common_starthour=df['hour'].value_counts().idxmax()
    
    #AM vs PM
    if most_common_starthour<=12:
        prt_time=str(most_common_starthour)+':00 AM'
    else:
        prt_time=str(most_common_starthour%12)+':00 PM'
    
    print('The most common start hour is: {}\n'.format(prt_time))


    print("\nThis took %s seconds." % (time.time() - start_time))
    print('-'*40)


def station_stats(df):
    """Displays statistics on the most popular stations and trip."""

    print('\nCalculating The Most Popular Stations and Trip...\n')
    start_time = time.time()

    # TO DO: display most commonly used start station
    most_common_startstation=df['Start Station'].value_counts().idxmax() 
    #print(df['Start Station'].value_counts().head())
    print('The most commonly used start station is: {}'.format(most_common_startstation))
 

    # TO DO: display most commonly used end station
    most_common_endstation=df['End Station'].value_counts().idxmax()
    #print(df['End Station'].value_counts().head())
    print('The most commonly used end station is: {}'.format(most_common_endstation))
 
    # TO DO: display most frequent combination of start station and end station trip
    most_common_StartandEndstation=(df['Start Station']+' --> '+df['End Station']).value_counts().idxmax()
    
    #print((df['Start Station']+' --> '+df['End Station']).value_counts().head())
    print('The most frequenct combination of start and end stations is: {}\n**This combination doesn\'t necessarily combine the most common start station and end station persay\n'.format(most_common_StartandEndstation))

    print("\nThis took %s seconds." % (time.time() - start_time))
    print('-'*40)


def trip_duration_stats(df):
    """Displays statistics on the total and average trip duration."""

    print('\nCalculating Trip Duration...\n')
    start_time = time.time()

    # TO DO: display total travel time
    duration_mean=df['Trip Duration'].mean()#.seconds
    data_count=df['Trip Duration'].count()
    duration_days=(duration_mean*data_count)//86400
    duration_hrs=((duration_mean*data_count)%86400)//3600
    duration_minutes=((duration_mean*data_count)%86400)%3600//60
    #print('seconds %f',duration_mean*data_count)
    print('The total time traveled is {} days, {} hrs, {} minutes. '.format(int(duration_days),int(duration_hrs),int(duration_minutes)))
    
    # TO DO: display mean travel time
    df['Subtracted_duration']=pd.to_datetime(df['End Time'])-pd.to_datetime(df['Start Time'])
    average_travelledtime=df['Subtracted_duration'].mean()
    days = average_travelledtime.days
    seconds=(average_travelledtime.seconds)%60
    minutes=(average_travelledtime.seconds)//60
    print('Mean travel time is {} days, {} hrs, {} minutes.\n'.format(days,minutes,seconds))
    

    print("\nThis took %s seconds." % (time.time() - start_time))
    print('-'*40)


def user_stats(df,city):
    """Displays statistics on bikeshare users."""

    print('\nCalculating User Stats...\n')
    start_time = time.time()

    # TO DO: Display counts of user types
    listtunique=df['User Type'].dropna().unique()
    print('Count of user types is {}:{}\n**Printing these types because in some cases a third type is present in the database\n'.format(len(df['User Type'].dropna().unique()),listtunique))


    #NOT VALID for washington
    if city != 'was':
        # TO DO: Display counts of gender
        print('Count of gender is {}'.format(len(df['Gender'].dropna().unique())))

        # TO DO: Display earliest, most recent, and most common year of birth
        earliest=int(df['Birth Year'].min())
        most_recent=int(df['Birth Year'].max())
        most_common=int(df['Birth Year'].mode())
        print('Earliest year of birth is {} \nMost recent year of birth is {}\nMost common year of birth is {}\n'.format(earliest,most_recent,most_common))
    else:
        print('Washington database doesn\'t contain gender and birth year information\n')
              

    print("\nThis took %s seconds." % (time.time() - start_time))
    print('-'*40)

# used to print raw data of city specified and check after printing whether to terminate or proceed to analyze stats    
def print_rawdata(stay_inraw):
    
    city = input("City abbreviation? chi,nyc,was:\n")
    while (city.lower() not in city_list):# and city.lower()!='stop'):
        print('Unexpected Input..')
        city=input("City abbreviation? chi,nyc,was:\n")
    df = pd.read_csv(CITY_DATA[city])
    print('number of data entries for {} is: {}\n'.format(city,len(df.index)))
    raw_data_counter=0#len(df.index)-12
    while stay_inraw=='yes'.lower():
        if raw_data_counter+5<=len(df.index):
            toprint=(raw_data_counter+5)
            print(df.iloc[raw_data_counter:toprint])
            #while wait:
                #try:
            stay_inraw = input('\nWould you like to see more raw data? Enter yes or no.\n')
            while (stay_inraw.lower() not in answers):# and city.lower()!='stop'):
                print('Unexpected Input..')
                stay_inraw=input('\nWould you like to see more raw data? Enter yes or no.\n')
        
            if stay_inraw=='yes'.lower():
                raw_data_counter+=5
        else:
            print('Data left to view is less than 5 entries!\n')
            stay_inraw='no'.lower()
            
            
        stay_in_program='yes'.lower()    
    
        
    
def main():
    global city_list,months_list,days_list,answers
    
    #lists used to navigate ValueErrors,global to not need to pass it to functions requiring it 
    city_list=['chi','nyc','was']
    months_list = ['january','february','march','april','may','june']
    days_list=['monday','tuesday','wednesday','thursday','friday','saturday','sunday']
    answers=['yes','no']
    
    #parameter reponsible for terminating the program
    stay_in_program = 'yes'
    
    try: #catch keyboard interrupe error by exception
        stay_inraw=input('Hey There! Would you like to see raw data? Enter yes or no.\n')

        while (stay_inraw.lower() not in answers):# and city.lower()!='stop'):
            print('Unexpected Input..') #ValueError combating
            stay_inraw=input('\nWould you like to see raw data? Enter yes or no.\n')

        while stay_in_program=='yes'.lower():
            if stay_inraw=='yes'.lower():
                print_rawdata(stay_inraw)
                stay_inraw='no'.lower()

            elif stay_inraw=='no'.lower():
                
                city,month,day = get_filters()
                df = load_data(city,month,day)
                
                time_stats(df)
                station_stats(df) 
                trip_duration_stats(df)    
                user_stats(df,city)
                
                stay_in_program = input('\nWould you like to keep exploring bikeshare stats? Enter yes or no.\n')
                while (stay_in_program.lower() not in answers):# and city.lower()!='stop'):
                    print('Unexpected Input..')#ValueError combating
                    stay_in_program=input('\nWould you like to keep exploring bikeshare stats? Enter yes or no.\n')
                if stay_in_program.lower() == 'no': 
                    break
    except KeyboardInterrupt:
        print ('\n Caught a KeyboardInterrupt!')

if __name__ == "__main__":
	main()
"""
Rubrics:
    1)code run with no errors 
    2)Error handeling 
    -typos->decide on valid values for each input, otherwise keep asking for input
    -lower/upper cases--> use lower() after any input to make input case insensetive
    -KeyboardInterrupt--> catch exception then terminate program
    3)comments and good code practice
    4)suitable data types,packages & loops
    5)**blanks in files replaced with na
    6)Stats correctly computed
    7)Raw data shown in fives when requested untill the user asks for no more or there is no more to show
    """