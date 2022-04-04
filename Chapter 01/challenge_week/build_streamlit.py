# imputing libraries

from tkinter.tix import COLUMN
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
from matplotlib.patches import Ellipse
import pandas as pd
import seaborn as sns
import numpy as np

#input data into dataframe

    
df_restaurant=pd.read_csv(r'./restaurant_data.csv') # restaurant 

df_pub=pd.read_csv(r'./pubs_data.csv') 

df_hotel=pd.read_csv(r'./hotels_data.csv')

mapper_res = {'£':1, '££':2, '£££':3, '££££':4}
df_restaurant['restaurant_price'] = df_restaurant['restaurant_price'].map(mapper_res)

mapper = {'£':1, '££':2, '£££':3, '££££':4}
df_hotel['hotel_price'] = df_hotel['hotel_price'].map(mapper)
df_pub['pub_price'] = df_pub['pub_price'].map(mapper)

pub_sum, pub_price_mean = df_pub['pub_reviews'].sum(), df_pub['pub_price'].mean()
res_sum, res_price_mean = df_restaurant['restaurant_reviews'].sum(), df_restaurant['restaurant_price'].mean()
hot_sum, hotel_price_mean = df_hotel['hotel_reviews'].sum(), df_hotel['hotel_price'].mean()

pub_sum1, pub_price_mean = df_pub['pub_reviews'].sum(), df_pub['pub_price'].mean()
res_sum1, res_price_mean = df_restaurant['restaurant_reviews'].sum(), df_restaurant['restaurant_price'].mean()
hot_sum1, hotel_price_mean = df_hotel['hotel_reviews'].sum(), df_hotel['hotel_price'].mean()


res1 = df_restaurant['restaurant_ratings'].mean()
hot1 = df_hotel['hotel_ratings'].mean()
pub1 = df_pub['pub_ratings'].mean()
res2 = df_restaurant['restaurant_reviews'].mean()
hot2 = df_hotel['hotel_reviews'].mean()
pub2 = df_pub['pub_reviews'].mean()


res_profit , res_main_profit= res_price_mean * res_sum, res_price_mean * res2

hotel_profit , hotel_main_profit= hotel_price_mean * hot_sum, hotel_price_mean * hot2

pub_profit, pub_main_profit = pub_price_mean * pub_sum, pub_price_mean * pub2

location = df_restaurant['restaurant_neighbourhoods'].value_counts()
loc1 = location.nlargest(10)



# objectives of work
def objectives():
    st.header('*The objectives of the project is to:* \n' )  
    st.markdown('### 1. Analysed Pubs, Hotels and Restaurants business in London \n' 

                '### 2. Present recommendation on what Business is more appealing to people in London')
        
# outline   
def outline():
    st.markdown('### 1. Project Objectives \n ' 

                '### 2. Methodology of the Work \n'

                '### 3. Analysis \n'

                '### 4. Main Analysis \n'

                '### 5. Final Result \n'

                '### 6. Recommendation')
        
    
# methods and show data
def methodology():
    st.markdown('#### Data used was London Hotels, Restaurants and Pubs, scraped from [yelp](https://www.yelp.co.uk/search?find_desc=&find_loc=London%2C+United+Kingdom&ns=1) website \n')
    #st.header("Converted Price into Number")
    st.markdown("### Converting Pounds:  £ = 1, ££ = 2, £££ = 3, ££££ = 4")

    st.markdown('### ***Restaurants Data***')
    st.dataframe(df_restaurant)

    st.markdown('### ***Pubs Data***')
    st.dataframe(df_pub)

    st.markdown('### ***Hotels Data***')
    st.dataframe(df_hotel)
    


#### Total Data mean

def bar_chart__total():
    labels = ['restaurants', 'hotels', 'pubs']
    rating_mean = [4.35, 3.96, 4.03]
    review_mean = [109.75, 30.78, 62.37]
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, rating_mean, width, label='Rating')
    rects2 = ax.bar(x + width/2, review_mean, width, label='Review')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_title('Comparing Rating and Review')
    ax.set_xticks(x, labels)
    ax.legend()
    st.pyplot(fig)



#### box plot

def box_chart__res():
    fig = plt.figure(figsize = (10, 5))
    sns.boxplot(x="restaurant_ratings", y="restaurant_reviews", data=df_restaurant)
    plt.ylabel("Restaurants reviews")
    plt.xlabel("Restaurants ratings")
    plt.title("Comparing Restaurants Ratings and Restaurants Reviews")
    st.pyplot(fig)

def box_chart__hot():
    fig = plt.figure(figsize = (10, 5))
    sns.boxplot(data=df_hotel, x="hotel_ratings", y="hotel_reviews")
    plt.ylabel("Hotels reviews")
    plt.xlabel("Hotels ratings")
    plt.title("Comparing Hotels Ratings and Hotels Reviews")
    st.pyplot(fig)


def box_chart__pub():
    fig = plt.figure(figsize = (10, 5))
    sns.boxplot(x="pub_ratings", y="pub_reviews", data=df_pub)
    plt.ylabel("Pubs reviews")
    plt.xlabel("Pubs ratings")
    plt.title("Comparing Pubs Ratings and Pubs Reviews")
    st.pyplot(fig)



#### Std


def std__res():
        a = df_restaurant["restaurant_ratings"]
        b = df_restaurant["restaurant_reviews"]
        c = df_restaurant["restaurant_price"]
        df = pd.DataFrame({'Restaurants Ratings': a, 'Restaurants Reviews': b, 'Restaurants Price': c})
        means = df.groupby('Restaurants Price').mean()
        sdevs = df.groupby('Restaurants Price').std()

        fig, axes = plt.subplots(ncols=2, figsize=(12, 4))
        colors = ['crimson', 'dodgerblue', 'limegreen', 'turquoise']
        for ax in axes:
            sns.scatterplot(x='Restaurants Ratings', y='Restaurants Reviews', hue='Restaurants Price', palette=colors, s=5, ec='none', data=df, ax=ax)
            sns.scatterplot(x='Restaurants Ratings', y='Restaurants Reviews', marker='o', s=50, fc='none', ec='black', label='means', data=means, ax=ax)
            if ax == axes[1]:
                for (_, mean), (_, sdev), color in zip(means.iterrows(), sdevs.iterrows(), colors):
                    for sdev_mult in [1, 2, 3]:
                        ellipse = Ellipse((mean['Restaurants Ratings'], mean['Restaurants Reviews']), width=2 * sdev['Restaurants Ratings'] * sdev_mult,
                                        height=2 * sdev['Restaurants Reviews'] * sdev_mult,
                                        facecolor=color, alpha=0.2 if sdev_mult == 1 else 0.1)
                        ax.add_patch(ellipse)
    
        st.pyplot(fig)


def std__hot():
    a = df_hotel["hotel_ratings"]
    b = df_hotel["hotel_reviews"]
    c = df_hotel["hotel_price"]
    df = pd.DataFrame({'Hotels Ratings': a, 'Hotels Reviews': b, 'Hotels Price': c})
    means = df.groupby('Hotels Price').mean()
    sdevs = df.groupby('Hotels Price').std()

    fig, axes = plt.subplots(ncols=2, figsize=(12, 4))
    colors = ['crimson', 'dodgerblue', 'limegreen', 'turquoise']
    for ax in axes:
        sns.scatterplot(x='Hotels Ratings', y='Hotels Reviews', hue='Hotels Price', palette=colors, s=5, ec='none', data=df, ax=ax)
        sns.scatterplot(x='Hotels Ratings', y='Hotels Reviews', marker='o', s=50, fc='none', ec='black', label='means', data=means, ax=ax)
        if ax == axes[1]:
            for (_, mean), (_, sdev), color in zip(means.iterrows(), sdevs.iterrows(), colors):
                for sdev_mult in [1, 2, 3]:
                    ellipse = Ellipse((mean['Hotels Ratings'], mean['Hotels Reviews']), width=2 * sdev['Hotels Ratings'] * sdev_mult,
                                    height=2 * sdev['Hotels Reviews'] * sdev_mult,
                                    facecolor=color, alpha=0.2 if sdev_mult == 1 else 0.1)
                    ax.add_patch(ellipse)
    st.pyplot(fig)


def std__pub():
    a = df_pub["pub_ratings"]
    b = df_pub["pub_reviews"]
    c = df_pub["pub_price"]
    df = pd.DataFrame({'Pubs Ratings': a, 'Pubs Reviews': b, 'Pubs Price': c})
    means = df.groupby('Pubs Price').mean()
    sdevs = df.groupby('Pubs Price').std()

    fig, axes = plt.subplots(ncols=2, figsize=(12, 4))
    colors = ['crimson', 'dodgerblue', 'limegreen', 'turquoise']
    for ax in axes:
        sns.scatterplot(x='Pubs Ratings', y='Pubs Reviews', hue='Pubs Price', palette=colors, s=5, ec='none', data=df, ax=ax)
        sns.scatterplot(x='Pubs Ratings', y='Pubs Reviews', marker='o', s=50, fc='none', ec='black', label='means', data=means, ax=ax)
        if ax == axes[1]:
            for (_, mean), (_, sdev), color in zip(means.iterrows(), sdevs.iterrows(), colors):
                for sdev_mult in [1, 2, 3]:
                    ellipse = Ellipse((mean['Pubs Ratings'], mean['Pubs Reviews']), width=2 * sdev['Pubs Ratings'] * sdev_mult,
                                    height=2 * sdev['Pubs Reviews'] * sdev_mult,
                                    facecolor=color, alpha=0.2 if sdev_mult == 1 else 0.1)
                    ax.add_patch(ellipse)
    st.pyplot(fig)

    


##### Profit in reviews by sum values


def bar_chart_reviews():
    #Creating the dataset
    fig = plt.figure(figsize = (10, 5))
    data = {'Restaurants':res_profit,'Hotels': hotel_profit, 'Pubs':pub_profit }
    Courses = list(data.keys())
    values = list(data.values())
    plt.subplot(1, 2, 1)
    plt.barh(Courses, values)
    plt.title("Average Reviews for profit")
    st.pyplot(fig)

def bar_chart_reviews1():
    #Creating the dataset
    fig = plt.figure(figsize = (10, 5))
    
    plt.subplot(1, 2, 2)
    labels = ['Restaurants', 'Hotels', 'Pubs']
    ratings = [res_profit, hotel_profit, pub_profit]
    explode = (0.1, 0, 0)  
    plt.title("Average Reviews for profit in Percentage")
    plt.pie(ratings, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
    plt.axis('equal')  # Equal aspect ratio ensu
    st.pyplot(fig)


### mean values for reviews


def barChart_profit():
    fig = plt.figure(figsize = (10, 5))
    data = {'Restaurants':res_main_profit,'Hotels' :hotel_main_profit , 'Pubs': pub_main_profit}
    Courses = list(data.keys())
    values = list(data.values())
    plt.subplot(1, 2, 1)
    plt.barh(Courses, values)
    plt.title('Profit of Business in London')
    st.pyplot(fig)

def barChart_profit1():
    fig = plt.figure(figsize = (10, 5))
   
    plt.subplot(1, 2, 2)
    labels = ['Restaurants', 'Hotels', 'Pubs']
    plt.title('Profit of Business in London in Percentage')
    profit= [res_main_profit, hotel_main_profit, pub_main_profit]
    explode = (0.1, 0, 0)  
    plt.pie(profit, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig)



##### Profit in ratings

res_profit1 , res_main_profit1= res_price_mean * res_sum1, res_price_mean * res1

hotel_profit1 , hotel_main_profit1= hotel_price_mean * hot_sum1, hotel_price_mean * hot1

pub_profit1, pub_main_profit1 = pub_price_mean * pub_sum1, pub_price_mean * pub1


def bar_chart_ratings():
    #Creating the dataset
    fig = plt.figure(figsize = (10, 5))
    data1 = {'Restaurants':res_profit1,'Hotels': hotel_profit1, 'Pubs':pub_profit1 }
    Courses1 = list(data1.keys())
    values1 = list(data1.values())
    plt.subplot(1, 2, 1)
    plt.barh(Courses1, values1)
    plt.title("Average Ratings for profit")
    st.pyplot(fig)

def bar_chart_ratings1():
    #Creating the dataset
    fig = plt.figure(figsize = (10, 5))
    
    plt.subplot(1, 2, 2)
    labels = ['Restaurants', 'Hotels', 'Pubs']
    reviews = [res_profit1, hotel_profit1, pub_profit1]
    explode = (0, 0.1, 0)  
    plt.title("Average Ratings for profit in Percentage")
    plt.pie(reviews, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
    plt.axis('equal')  # Equal aspect ratio ensu
    st.pyplot(fig)

### mean in ratings


def barChart_profit_ratings():
    fig = plt.figure(figsize = (10, 5))
    data = {'Restaurants':res_main_profit1,'Hotels' :hotel_main_profit1 , 'Pubs': pub_main_profit1}
    Courses = list(data.keys())
    values = list(data.values())
    plt.subplot(1, 2, 1)
    plt.barh(Courses, values)
    plt.title('Profit of Business in London')
    st.pyplot(fig)

def barChart_profit_ratings1():
    fig = plt.figure(figsize = (10, 5))
   
    plt.subplot(1, 2, 2)
    labels = ['Restaurants', 'Hotels', 'Pubs']
    plt.title('Profit of Business in London in Percentage')
   
    profit= [res_main_profit1, hotel_main_profit1, pub_main_profit1]
    explode = (0, 0.1, 0)  
    plt.pie(profit, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig)



##### Average ratings

def bar_chart_for_average_ratings():
    #Creating the dataset
    fig = plt.figure(figsize = (10, 5))
    data = {'Restaurants':res1,'Hotels': hot1, 'Pubs':pub1 }
    Courses = list(data.keys())
    values = list(data.values())
    plt.subplot(1, 2, 1)
    plt.barh(Courses, values)
    plt.title("Average Ratings")
    st.pyplot(fig)

def bar_chart_for_average_ratings1():
    fig = plt.figure(figsize = (10, 5))    
    plt.subplot(1, 2, 2)
    labels = ['Restaurants', 'Hotels', 'Pubs']
    ratings = [res1, hot1, pub1]
    explode = (0.1, 0, 0)  
    plt.title("Average Ratings in Percentage")
    plt.pie(ratings, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
    plt.axis('equal')  # Equal aspect ratio ensu
    st.pyplot(fig)


#### Average reviews


def bar_chart_for_average_reviews():
    #Creating the dataset
    fig = plt.figure(figsize = (10, 5))
    data = {'Restaurants':res2,'Hotels': hot2, 'Pubs':pub2}
    Courses = list(data.keys())
    values = list(data.values())
    plt.subplot(1, 2, 1)
    plt.barh(Courses, values)
    plt.title("Average Reviews")
    st.pyplot(fig)

def bar_chart_for_average_reviews1():
    fig = plt.figure(figsize = (10, 5))    
    plt.subplot(1, 2, 2)
    labels = ['Restaurants', 'Hotels', 'Pubs']
    ratings = [res2, hot2, pub2]
    explode = (0.1, 0, 0)  
    plt.title("Average Reviews in Percentage")
    plt.pie(ratings, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
    plt.axis('equal')  # Equal aspect ratio ensu
    st.pyplot(fig)


##### mostly occupied area

def bar_chart_location():
    location = df_restaurant['restaurant_neighbourhoods'].value_counts()
    loc1 = location.nlargest(10)
    name = ['Soho', 'Mayfair', 'Covent Garden', 'Fitzrovia', 'London Bridge', 'Bloomsbury', 'Euston', 'Chelsea', 'Notting Hill', 'Borough']
    area = pd.DataFrame({'Location Name': name, 'Total Number': loc1})

    fig = px.bar(
                area, x = "Location Name", y = "Total Number",
                                template = 'seaborn',
                                title = '', 
                                
                                        )

    st.plotly_chart(fig) 


#### most food

def bar_chart_food():
    food = df_restaurant['restaurant_type'].value_counts()
    cuisine = food.nlargest(10)
    name = ['Italian', 'British', 'PubsBritish', 'Japanese', 'BritishBars', 'BritishModern European', 'Caribbean', 'Cafes', 'Breakfast & Brunch', 'French']
    area = pd.DataFrame({'Cuisines Name': name, 'Total Number': cuisine})

    fig = px.bar(
                area, x = "Cuisines Name", y = "Total Number",
                                template = 'seaborn',
                                title = 'Top Cuisines in worst case', 
                                
                                        )

    st.plotly_chart(fig) 


#### best location

best_option = df_restaurant[(df_restaurant['restaurant_ratings'] > res1) & (df_restaurant['restaurant_reviews'] > res2)]
best = best_option.head(10)

def bar_chart_best_location():
    best_loc = best_option["restaurant_neighbourhoods"].value_counts()
    loc_best = best_loc.nlargest(10)
    name3 = ['Soho', 'Bloomsbury', 'Kensington', 'Covent Garden', 'Islington', 'Victoria', 'West Brompton', 'Mayfair', 'Fitzrovia', 'Blackfriars']
    area = pd.DataFrame({'Location Name': name3, 'Total Number': loc_best})

    fig = px.bar(
                area, x = "Location Name", y = "Total Number",
                                template = 'seaborn',
                                title = 'Top Rated 10 Location', 
                                
                                        )

    st.plotly_chart(fig) 


#### best food

def bar_chart_best_food():
    best_res = best_option["restaurant_type"].value_counts()
    res_best = best_res.nlargest(10)
    name2 = ['British', 'Fish & Chips', 'GastropubsBritish', 'Italian', 'Indian', 'BritishGastropubs', 'French', 'Pizza', 'BritishCocktail BarsSteakhouses', 'Steakhouses']
    area = pd.DataFrame({'Cuisines Name': name2, 'Total Number': res_best})

    fig = px.bar(
                area, x = "Cuisines Name", y = "Total Number",
                                template = 'seaborn',
                                title = 'Top Rated 10 Cuisines', 
                                
                                        )

    st.plotly_chart(fig) 



#### compare

location = df_restaurant['restaurant_neighbourhoods'].value_counts()
loc1 = location.nlargest(10)

def pie_chart_location():

        labels1 = ['Soho', 'Mayfair', 'Covent Garden', 'Fitzrovia', 'London Bridge']
        explode1 = (0.1, 0, 0, 0, 0)

        labels2 = 'Soho', 'Bloomsbury', 'Kensington', 'Covent Garden', 'Islington'
        explode2 = (0.1, 0, 0, 0, 0)

        fig1, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))

        ax1.pie(loc1, explode=explode1, labels=labels1, shadow=True, startangle=90)
        ax2.pie(loc1, explode=explode2, labels=labels2, shadow=True, startangle=90)

        ax1.set_xlabel('Bad Data', size=15)
        ax2.set_xlabel('Best Data', size=15)

        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        ax2.axis('equal')

        plt.tight_layout()
        st.pyplot(fig1)


#### compare food


def pie_chart_food():

    labels3 = 'Italian', 'British', 'PubsBritish', 'Japanese', 'BritishBars'
    explode3 = (0.1, 0, 0, 0, 0)

    labels4 = 'British', 'Fish & Chips', 'GastropubsBritish', 'Italian', 'Indian'
    explode4 = (0.1, 0, 0, 0, 0)

    fig1, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))

    ax1.pie(loc1, explode=explode3, labels=labels3, shadow=True, startangle=90)
    ax2.pie(loc1, explode=explode4, labels=labels4, shadow=True, startangle=90)

    ax1.set_xlabel('Bad Data', size=15)
    ax2.set_xlabel('Best Data', size=15)

    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax2.axis('equal')
    
    plt.tight_layout()
    st.pyplot(fig1)





def analysis():
    ratings = [res1, hot1, pub1]
    st.bar_char(
            ['restaurant', 'hotel', 'pub'], ratings
    )
    
    pass


      
def recommeded_analysis():
    
    pass




#  Output in the Streamlit App.
def main():
    #st.title('Kojo')
    page = st.sidebar.selectbox(
        "Main Content", 
        [
            "Title",
            "Presentation Outline",
            "Objectives",
            "Methodology",
            "Analysis",
            "Analysis of Results",
            "Final Result",
            "Recommendation"
        ],
        
    )
    
    if page=='Title':
       st.title("Most popular business in London by opinion customers")
       st.image("Downloads\\london.jpg", use_column_width = True)
    

    #First Page
    elif page == "Presentation Outline":
        st.title("Presentation Outline")
        st.image("Downloads\\pre1.jpg", use_column_width = True)
        outline()


    #Second Page
    elif page == "Objectives":
       
        objectives()
        st.image("Downloads\\food.jpg", use_column_width = True)
        
    
    #Third Page
    elif page == "Methodology":

        methodology()
        

    #Fourth Page
    elif page == "Analysis":

        
         page1 = st.sidebar.selectbox(
             "Sub Content", 
            [
                "Bar Chart for Average Reviews",
                "Bar Chart for Average Ratings",
                "Bar Chart for Total Average",
                "Box Plot for Restaurants, Hotels and Pubs",
                "STD for Restaurants Price,  Hotels Price and Pubs Price",
                "Bar Chart for Reviews in Sum",
                "Bar Chart for Reviews in Mean",
                "Bar Chart for Ratings in Sum",
                "Bar Chart for Ratings in Mean" 
            ],
            
        )

         
         if  page1 == "Bar Chart for Average Reviews":
                st.title("Bar and Pie Chart for Average Reviews")
                bar_chart_for_average_reviews()
                bar_chart_for_average_reviews1()

         elif page1 == "Bar Chart for Average Ratings":
                st.title("Bar and Pie Chart for Average Ratings")
                bar_chart_for_average_ratings()
                bar_chart_for_average_ratings1()
            
         elif page1 == "Bar Chart for Total Average":
                st.title("Bar Chart for Total Average")
                bar_chart__total()

         elif page1 == "Box Plot for Restaurants, Hotels and Pubs":
               st.title("Box Plot for Restaurants, Hotels and Pubs")
               box_chart__res()
               box_chart__hot()
               box_chart__pub()

         elif page1 == "STD for Restaurants Price,  Hotels Price and Pubs Price":
                st.title("Mean and STD for Restaurants Price,  Hotels Price and Pubs Price")
                std__res()
                std__hot()
                std__pub()

         elif page1 == "Bar Chart for Reviews in Sum":
                st.title("Bar and Pie Chart for Reviews in Sum")
                bar_chart_reviews()
                bar_chart_reviews1()

         elif page1 == "Bar Chart for Reviews in Mean":
                st.title("Bar and Pie Chart for Reviews in Mean")
                barChart_profit()
                barChart_profit1()

         elif page1 == "Bar Chart for Ratings in Sum":
                st.title("Bar and Pie Chart for Ratings in Sum")
                bar_chart_ratings()
                bar_chart_ratings1()

         elif page1 == "Bar Chart for Ratings in Mean":
                st.title("Bar and Pie Chart for Ratings in Mean") 
                barChart_profit_ratings()
                barChart_profit_ratings1()



    # fifth page
    elif page == "Analysis of Results":
           
           page2 = st.sidebar.selectbox(
             "Sub Content", 
            [
                "Top Location in worst case",
                "Top Cuisines in worst case",
                "Top Rated 10 Location",
                "Top Rated 10 Cuisines"
            ],
            
        )

           if page2 == "Top Location in worst case":
               st.title("Top Location in worst case")
               bar_chart_location()

           elif page2 == "Top Cuisines in worst case":
               st.title("Top Cuisines in worst case")
               bar_chart_food()

           elif page2 == "Top Rated 10 Location":
                  st.title("Top Rated 10 Location")
                  bar_chart_best_location()

           elif page2 == "Top Rated 10 Cuisines":
                  st.title("Top Rated 10 Cuisines")
                  bar_chart_best_food()


    #Fifth page
    elif page == "Final Result":
        st.title("Final Result")
        st.header("Restaurants")
        st.image("Downloads\\res.jpg", use_column_width = True)
        st.balloons()


    elif page == "Recommendation":
        st.title("Recommended Location")
        pie_chart_location()
        st.title("Recommended Cuisines")
        pie_chart_food()

    
    
    
        
    

if __name__ == "__main__":
    main()