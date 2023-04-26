import pandas as pd
import altair as alt
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA


# With the use of Altair and Plotly charts in a Streamlit app, this code displays a collection of interactive visualizations for investigating data on worldwide disasters. Charts showing the quantity and types of catastrophes can be viewed by choosing a country, a year, or both. The visualizations offer a clear, engaging, and interactive method to comprehend patterns and occurrences in major global disasters.


def page_all_disasters():

    df = pd.read_csv("./data/Main.csv")
    countries = df['Country'].unique()

    st.write(f"## Total disasters for a specific country")

    selected_country1 = st.selectbox("Select a country for chart 1", countries, key='chart1')
    country_data = df[df['Country'] == selected_country1].drop('Total', axis=1)
    melted_data = pd.melt(country_data, id_vars=['ObjectId', 'Country', 'Indicator'], var_name='Year', value_name='Total')
    chart1 = alt.Chart(melted_data[melted_data['Indicator'] != 'TOTAL']).mark_bar().encode(
        x=alt.X('Year:N', title='Year'),
        y=alt.Y('Total:Q', title='Total'),
        color='Indicator:N',
        tooltip=['Year', 'Total', 'Indicator']
    ).properties(
        width=800,
        height=500,
        title=f"Country - {selected_country1}"
    )
    st.altair_chart(chart1)

    st.write(f"## Trend of total disasters for a specific country")

    selected_country2 = st.selectbox("Select a country for chart 1", countries, key='chart2')
    country_data = df[df['Country'] == selected_country2].drop('Total', axis=1)
    melted_data = pd.melt(country_data, id_vars=['ObjectId', 'Country', 'Indicator'], var_name='Year', value_name='Total')
    melted_data = melted_data[melted_data['Indicator'] != 'TOTAL']
    chart2 = alt.Chart(melted_data).mark_line().encode(
        x=alt.X('Year:N', title='Year'),
        y=alt.Y('Total:Q', title='Total'),
        color='Indicator:N',
        tooltip=['Year', 'Total', 'Indicator']
    ).properties(
        width=800,
        height=500,
        title=f"Country -  {selected_country2}"
    )
    st.altair_chart(chart2)

    st.write(f"## Number of Disasters in a Selected Country Over the Last Two Decades")

    df = pd.read_csv("./data/Main.csv")
    countries = df['Country'].unique()
    selected_country = st.selectbox("Select a country", countries, key='country_select')
    country_data = df[df['Country'] == selected_country]
    country_data = country_data[country_data['Indicator'] != 'TOTAL']
    bar_chart = alt.Chart(country_data).mark_bar().encode(
        x=alt.X('Indicator:N', sort='-x'),
        y=alt.Y('Total:Q', axis=alt.Axis(title='Occurrences')),
        tooltip=['Indicator', 'Total']
    ).properties(
        width=300,
        height=200,
    )
    pie_chart = alt.Chart(country_data).mark_arc().encode(
        theta='Total:Q',
        color='Indicator:N',
        tooltip=['Indicator', 'Total']
    ).properties(
        width=300,
        height=200,
    )
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write(alt.hconcat(bar_chart, pie_chart))

    df = pd.read_csv("./data/Main.csv")
    years = [str(year) for year in range(2001, 2022)]

    st.write(f"## Distribution of types of disasters across all countries for a specific year")

    selected_year = st.selectbox("Select a year", years)
    year_data = df[['Country', 'Indicator', selected_year]]
    grouped_data = year_data.groupby('Indicator').sum().reset_index()
    grouped_data = grouped_data[grouped_data['Indicator'] != 'TOTAL']
    chart = alt.Chart(grouped_data).mark_arc().encode(
        theta=selected_year,
        color='Indicator:N',
        tooltip=['Indicator', selected_year]
    ).properties(
        width=700,
        height=400,
        title=f"Distribution of types of disasters across all countries in {selected_year}"
    )
    st.altair_chart(chart)
    total = year_data[selected_year].sum()
    st.write(f"Total occurrences of all types of disasters in all countries in {selected_year}: {total}")
    
    total_data = df[df['Indicator'] == 'TOTAL'].reset_index()
    st.write(f"## Total occurrences of disasters by country")
    map_data = total_data.groupby('Country')['Total'].sum().reset_index()
    fig = px.choropleth(map_data, locations='Country', locationmode='country names',
                        color='Total', range_color=(0, map_data['Total'].max()),
                        width=800, height=600)
    st.plotly_chart(fig)
    
    df_original = pd.read_csv("./data/Original.csv")

    df_cleaned = pd.read_csv("./data/Main.csv")

    selected_dataset = st.radio("Select dataset", ("Original", "Cleaned"))
    if selected_dataset == "Original":
        st.write("# Original Dataset")
        st.write(df_original)
    else:
        st.write("# Cleaned Dataset")
        st.write(df_cleaned)

    
# In order to predict the likelihood of natural catastrophes in a chosen country and disaster type over the course of the following five years, this code implements an ARIMA model. A button is clicked to generate the forecast after the user chooses the country and kind of disaster from a selection menu. Using an Altair line chart, the predictions are shown.


def prediction():
  
    def fit_and_forecast_arima(data, country, indicator):
        filtered_data = data[(data['Country'] == country) & (data['Indicator'] == indicator)]
        filtered_data = filtered_data[years].T
        filtered_data.index = pd.to_datetime(filtered_data.index, format='%Y')

        try:
            arima_model = ARIMA(filtered_data, order=(1, 1, 1))
            arima_results = arima_model.fit()
            forecast_arima = arima_results.forecast(steps=5)
        except ValueError:
            forecast_arima = pd.Series([0] * 5, index=pd.date_range(start=filtered_data.index[-1] + pd.DateOffset(years=1), periods=5, freq='AS'))

        return forecast_arima

    data = pd.read_csv('./data/Main.csv')
    years = [str(x) for x in range(2001, 2022)]

    st.title('Natural Disaster Prediction')
    st.write('Select a country and disaster type to forecast occurrences in the next 5 years.')

    countries = data['Country'].unique().tolist()
    disasters = data[data['Indicator'] != 'TOTAL']['Indicator'].unique().tolist()
    selected_country = st.selectbox('Country:', countries, index=countries.index('United States'))
    selected_disaster = st.selectbox('Disaster Type:', disasters, index=disasters.index('Storm'))

    if st.button('Get Prediction'):
        forecast_arima = fit_and_forecast_arima(data, selected_country, selected_disaster)

        st.subheader(f'ARIMA Predictions for {selected_country} - {selected_disaster}')
        chart_data = pd.DataFrame({
            'Year': forecast_arima.index.year,
            'Predictions': forecast_arima.values
        })

        chart = alt.Chart(chart_data).mark_line().encode(
            alt.X('Year:O', axis=alt.Axis(title='Year')),
            alt.Y('Predictions:Q', axis=alt.Axis(title='Predictions'))
        )

        st.altair_chart(chart, use_container_width=True)

        
# With the help of this code, you may explore drought data interactively, including the frequency and number of droughts by nation and year. In addition to viewing a bar chart of frequency through time, users may also explore additional charts like a choropleth map, a bubble chart, and a pie chart by choosing countries from a dropdown menu. An accessible way to comprehend patterns and trends in drought data is made available by these visuals.


def page_second():

    df = pd.read_csv("./data/Drought.csv")
    
    countries = df['Country'].unique()
    
    st.write("# Drought Frequency by Country")

    selected_countries = st.multiselect("Select countries", countries, default=["United States", "India"])

    country_data = df[df['Country'].isin(selected_countries)].drop(['Total','ObjectId'], axis=1)

    melted_data = pd.melt(country_data, id_vars=['Country', 'Indicator'], var_name='Year', value_name='Drought Frequency')

    chart = alt.Chart(melted_data[melted_data['Indicator'] == 'Drought']).mark_bar().encode(
        x=alt.X('Year:N', title='Year', axis=alt.Axis(labelAngle=-45)),
        y=alt.Y('Drought Frequency:Q', title='Drought Frequency'),
        color=alt.Color('Country:N', legend=alt.Legend(title="Country")),
    ).properties(
        width=800,
        height=500,
        title="Drought Frequency by Country"
    )

    st.altair_chart(chart)
    
    ###############################################################
    
    st.write(f"### Geographical Distribution of Drought Occurrences Among Various Countries")
    fig = px.choropleth(data_frame=df,
                    locations='Country',
                    locationmode='country names',
                    color='Total',
                    scope='world')

    st.plotly_chart(fig)

    ###############################################################

    st.write(f"## Drought Count by Year for a Specific Country")

    selected_country = st.selectbox("Select a Country", countries, key='chart2')

    country_data = df[df['Country'] == selected_country].drop(['Total', 'Indicator', 'ObjectId'], axis=1)

    melted_data = pd.melt(country_data, id_vars=['Country'], var_name='Year', value_name='Drought_Count')

    chart2 = alt.Chart(melted_data).mark_bar(color='brown').encode(
        x=alt.X('Year:N', title='Year'),
        y=alt.Y('Drought_Count:Q', title='Drought Count'),
        tooltip=['Year', 'Drought_Count']
    ).properties(
        width=600,
        height=500,
        title=f"Country selected - {selected_country}"
    )

    st.altair_chart(chart2)
    
    ###############################################################

    drought_data = df[df['Indicator'] == 'Drought'].groupby(['Country']).sum().reset_index()

    chart = alt.Chart(drought_data).mark_circle().encode(
        x=alt.X('Country:N', sort='-y'),
        y=alt.Y('Total:Q', title='Total Number of Droughts'),
        color=alt.Color('Country:N', legend=None),
        size=alt.Size('Total:Q', legend=None),
        tooltip=['Country', 'Total']
    ).properties(
        width=700,
        height=500
    ).interactive()
    
    st.write(f"### Proportion of Total Number of Droughts by Country")

    st.altair_chart(chart)
    
    ###############################################################

    data = pd.read_csv('./data/Drought.csv')

    total_occurrences = data["Total"].sum()

    year_columns = [str(year) for year in range(2001, 2022)]
    percentages = [data[year].sum() / total_occurrences * 100 for year in year_columns]

    df = pd.DataFrame({"Year": year_columns, "Percentage": percentages})

    st.write(f"### Contribution of Each Year's Drought Occurrences to the Total Number of Droughts")

    fig = px.pie(df, values="Percentage", names="Year")

    st.plotly_chart(fig)
    
    
# This code offers a collection of interactive visualizations for investigating data on severe temperatures, including the frequency and count of extreme temperatures by nation and year. Users can choose nations from a dropdown menu to view a bar chart showing frequency through time and can also explore additional charts including a choropleth map, a bubble chart, and a pie chart. The patterns and trends in the data on severe temperatures can be easily understood using these graphics.

    
def page_third():
    
    df = pd.read_csv("./data/Extreme_temperature.csv")

    countries = df['Country'].unique()

    st.write("# Extreme Temperature Frequency by Country")

    selected_countries = st.multiselect("Select countries", countries, default=["United States", "India"])

    country_data = df[df['Country'].isin(selected_countries)].drop(['Total','ObjectId'], axis=1)

    melted_data = pd.melt(country_data, id_vars=['Country', 'Indicator'], var_name='Year', value_name='Frequency')

    chart = alt.Chart(melted_data[melted_data['Indicator'] == 'Extreme temperature']).mark_bar().encode(
        x=alt.X('Year:N', title='Year', axis=alt.Axis(labelAngle=-45)),
        y=alt.Y('Frequency:Q', title='Frequency'),
        color=alt.Color('Country:N', legend=alt.Legend(title="Country")),
    ).properties(
        width=800,
        height=500,
        title="Extreme Temperature Frequency by Country"
    )

    st.altair_chart(chart)

    ###############################################################
    
    st.write(f"### Geographical Distribution of Extreme Temperature Occurrences Among Various Countries")
    fig = px.choropleth(data_frame=df,
                    locations='Country',
                    locationmode='country names',
                    color='Total',
                    scope='world')

    st.plotly_chart(fig)

    ###############################################################

    st.write(f"## Extreme Temperature Count by Year for a Specific Country")

    selected_country = st.selectbox("Select a Country", countries, key='chart2')

    country_data = df[df['Country'] == selected_country].drop(['Total', 'Indicator', 'ObjectId'], axis=1)

    melted_data = pd.melt(country_data, id_vars=['Country'], var_name='Year', value_name='Count')

    chart2 = alt.Chart(melted_data).mark_bar(color='red').encode(
        x=alt.X('Year:N', title='Year'),
        y=alt.Y('Count:Q', title='Count'),
        tooltip=['Year', 'Count']
    ).properties(
        width=600,
        height=500,
        title=f"Country selected - {selected_country}"
    )

    st.altair_chart(chart2)

    ###############################################################

    temperature_data = df[df['Indicator'] == 'Extreme temperature'].groupby(['Country']).sum().reset_index()

    chart = alt.Chart(temperature_data).mark_circle().encode(
        x=alt.X('Country:N', sort='-y'),
        y=alt.Y('Total:Q', title='Total Number of Extreme Temperatures'),
        color=alt.Color('Country:N', legend=None),
        size=alt.Size('Total:Q', legend=None),
        tooltip=['Country', 'Total']
    ).properties(
        width=700,
        height=500
    ).interactive()

    st.write(f"### Proportion of Total Number of Extreme Temperatures by Country")

    st.altair_chart(chart)

    ###############################################################

    data = pd.read_csv('./data/Extreme_temperature.csv')

    total_occurrences = data["Total"].sum()

    year_columns = [str(year) for year in range(2001, 2022)]
    percentages = [data[year].sum() / total_occurrences * 100 for year in year_columns]

    df = pd.DataFrame({"Year": year_columns, "Percentage": percentages})

    st.write(f"### Contribution of Each Year's Extreme Temperature Occurrences to the Total Number of Extreme Temperatures")

    fig = px.pie(df, values="Percentage", names="Year")

    st.plotly_chart(fig)
    

# This program defines the "page_fourth" function, which loads and displays flood statistics by nation. The frequency and number of floods in various nations are shown via interactive visualizations made with Plotly and Altair, such as bar charts and choropleth maps. Another feature of the function is a pie chart that displays the percentage contribution of each year to the overall frequency of the flood indicator events.


def page_fourth():
    
    df = pd.read_csv("./data/Flood.csv")

    countries = df['Country'].unique()

    st.write("# Flood Frequency by Country")

    selected_countries = st.multiselect("Select countries", countries, default=["United States", "India"])

    country_data = df[df['Country'].isin(selected_countries)].drop(['Total','ObjectId'], axis=1)

    melted_data = pd.melt(country_data, id_vars=['Country', 'Indicator'], var_name='Year', value_name='Flood Frequency')

    chart = alt.Chart(melted_data[melted_data['Indicator'] == 'Flood']).mark_bar().encode(
        x=alt.X('Year:N', title='Year', axis=alt.Axis(labelAngle=-45)),
        y=alt.Y('Flood Frequency:Q', title='Flood Frequency'),
        color=alt.Color('Country:N', legend=alt.Legend(title="Country")),
    ).properties(
        width=800,
        height=500,
        title="Flood Frequency by Country"
    )

    st.altair_chart(chart)

    ###############################################################

    st.write(f"### Geographical Distribution of Flood Occurrences Among Various Countries")
    fig = px.choropleth(data_frame=df,
                        locations='Country',
                        locationmode='country names',
                        color='Total',
                        scope='world')

    st.plotly_chart(fig)

    ###############################################################

    st.write(f"## Flood Count by Year for a Specific Country")

    selected_country = st.selectbox("Select a Country", countries, key='chart2')

    country_data = df[df['Country'] == selected_country].drop(['Total', 'Indicator', 'ObjectId'], axis=1)

    melted_data = pd.melt(country_data, id_vars=['Country'], var_name='Year', value_name='Flood_Count')

    chart2 = alt.Chart(melted_data).mark_bar(color='blue').encode(
        x=alt.X('Year:N', title='Year'),
        y=alt.Y('Flood_Count:Q', title='Flood Count'),
        tooltip=['Year', 'Flood_Count']
    ).properties(
        width=600,
        height=500,
        title=f"Country selected - {selected_country}"
    )

    st.altair_chart(chart2)

    ###############################################################

    flood_data = df[df['Indicator'] == 'Flood'].groupby(['Country']).sum().reset_index()

    chart = alt.Chart(flood_data).mark_circle().encode(
        x=alt.X('Country:N', sort='-y'),
        y=alt.Y('Total:Q', title='Total Number of Floods'),
        color=alt.Color('Country:N', legend=None),
        size=alt.Size('Total:Q', legend=None),
        tooltip=['Country', 'Total']
    ).properties(
        width=700,
        height=500
    ).interactive()

    st.write(f"### Proportion of Total Number of Floods by Country")

    st.altair_chart(chart)

    ###############################################################

    data = pd.read_csv('./data/Flood.csv')

    total_occurrences = data["Total"].sum()

    year_columns = [str(year) for year in range(2001, 2022)]
    percentages = [data[year].sum() / total_occurrences * 100 for year in year_columns]

    df = pd.DataFrame({"Year": year_columns, "Percentage": percentages})

    st.write(f"### Contribution of Each Year's Flood Occurrences to the Total Number of Floods")

    fig = px.pie(df, values="Percentage", names="Year")

    st.plotly_chart(fig)
    

# This website application's fifth page examines natural disasters. The program analyzes landslide data, creates charts showing the frequency of landslides by country and year, and uses a choropleth map to depict the distribution of landslide events among nations. Using pie charts, it also displays the total number of landslides by country and the percentage that each year contributes to the overall total.

    
def page_fifth():
    
    df = pd.read_csv("./data/Landslide.csv")

    countries = df['Country'].unique()

    st.write("# Landslide Frequency by Country")

    selected_countries = st.multiselect("Select countries", countries, default=["United States", "India"])

    country_data = df[df['Country'].isin(selected_countries)].drop(['Total','ObjectId'], axis=1)

    melted_data = pd.melt(country_data, id_vars=['Country', 'Indicator'], var_name='Year', value_name='Landslide Frequency')

    chart = alt.Chart(melted_data[melted_data['Indicator'] == 'Landslide']).mark_bar().encode(
        x=alt.X('Year:N', title='Year', axis=alt.Axis(labelAngle=-45)),
        y=alt.Y('Landslide Frequency:Q', title='Landslide Frequency'),
        color=alt.Color('Country:N', legend=alt.Legend(title="Country")),
    ).properties(
        width=800,
        height=500,
        title="Landslide Frequency by Country"
    )

    st.altair_chart(chart)

    ###############################################################

    st.write(f"### Geographical Distribution of Landslide Occurrences Among Various Countries")
    fig = px.choropleth(data_frame=df,
                    locations='Country',
                    locationmode='country names',
                    color='Total',
                    scope='world')

    st.plotly_chart(fig)

    ###############################################################

    st.write(f"## Landslide Count by Year for a Specific Country")

    selected_country = st.selectbox("Select a Country", countries, key='chart2')

    country_data = df[df['Country'] == selected_country].drop(['Total', 'Indicator', 'ObjectId'], axis=1)

    melted_data = pd.melt(country_data, id_vars=['Country'], var_name='Year', value_name='Landslide_Count')

    chart2 = alt.Chart(melted_data).mark_bar(color='yellow').encode(
        x=alt.X('Year:N', title='Year'),
        y=alt.Y('Landslide_Count:Q', title='Landslide Count'),
        tooltip=['Year', 'Landslide_Count']
    ).properties(
        width=600,
        height=500,
        title=f"Country selected - {selected_country}"
    )

    st.altair_chart(chart2)

    ###############################################################

    landslide_data = df[df['Indicator'] == 'Landslide'].groupby(['Country']).sum().reset_index()

    chart = alt.Chart(landslide_data).mark_circle().encode(
        x=alt.X('Country:N', sort='-y'),
        y=alt.Y('Total:Q', title='Total Number of Landslides'),
        color=alt.Color('Country:N', legend=None),
        size=alt.Size('Total:Q', legend=None),
        tooltip=['Country', 'Total']
    ).properties(
        width=700,
        height=500
    ).interactive()

    st.write(f"### Proportion of Total Number of Landslides by Country")

    st.altair_chart(chart)

    ###############################################################

    data = pd.read_csv('./data/Landslide.csv')

    total_occurrences = data["Total"].sum()

    year_columns = [str(year) for year in range(2001, 2022)]
    percentages = [data[year].sum() / total_occurrences * 100 for year in year_columns]

    df = pd.DataFrame({"Year": year_columns, "Percentage": percentages})

    st.write(f"### Contribution of Each Year's Landslide Occurrences to the Total Number of Landslides")

    fig = px.pie(df, values="Percentage", names="Year")

    st.plotly_chart(fig)

    
# The page_sixth function extracts information on storm frequency by nation from a CSV file and shows it in a number of charts. Users can browse storm frequency and count data by year while selecting one or more countries. The function also contains a chart displaying the percentage of total storms by country and a choropleth map of the storms' locations.

    
def page_sixth():


    df = pd.read_csv("./data/Storm.csv")

    countries = df['Country'].unique()

    st.write("# Storm Frequency by Country")

    selected_countries = st.multiselect("Select countries", countries, default=["United States", "India"])

    country_data = df[df['Country'].isin(selected_countries)].drop(['Total','ObjectId'], axis=1)

    melted_data = pd.melt(country_data, id_vars=['Country', 'Indicator'], var_name='Year', value_name='Storm Frequency')

    chart = alt.Chart(melted_data[melted_data['Indicator'] == 'Storm']).mark_bar().encode(
        x=alt.X('Year:N', title='Year', axis=alt.Axis(labelAngle=-45)),
        y=alt.Y('Storm Frequency:Q', title='Storm Frequency'),
        color=alt.Color('Country:N', legend=alt.Legend(title="Country")),
    ).properties(
        width=800,
        height=500,
        title="Storm Frequency by Country"
    )

    st.altair_chart(chart)

    ###############################################################

    st.write(f"### Geographical Distribution of Storm Occurrences Among Various Countries")
    fig = px.choropleth(data_frame=df,
                    locations='Country',
                    locationmode='country names',
                    color='Total',
                    scope='world')

    st.plotly_chart(fig)

    ###############################################################

    st.write(f"## Storm Count by Year for a Specific Country")

    selected_country = st.selectbox("Select a Country", countries, key='chart2')

    country_data = df[df['Country'] == selected_country].drop(['Total', 'Indicator', 'ObjectId'], axis=1)

    melted_data = pd.melt(country_data, id_vars=['Country'], var_name='Year', value_name='Storm_Count')

    chart2 = alt.Chart(melted_data).mark_bar(color='purple').encode(
        x=alt.X('Year:N', title='Year'),
        y=alt.Y('Storm_Count:Q', title='Storm Count'),
        tooltip=['Year', 'Storm_Count']
    ).properties(
        width=600,
        height=500,
        title=f"Country selected - {selected_country}"
    )

    st.altair_chart(chart2)

    ###############################################################

    storm_data = df[df['Indicator'] == 'Storm'].groupby(['Country']).sum().reset_index()

    chart = alt.Chart(storm_data).mark_circle().encode(
        x=alt.X('Country:N', sort='-y'),
        y=alt.Y('Total:Q', title='Total Number of Storms'),
        color=alt.Color('Country:N', legend=None),
        size=alt.Size('Total:Q', legend=None),
        tooltip=['Country', 'Total']
    ).properties(
        width=700,
        height=500
    ).interactive()

    st.write(f"### Proportion of Total Number of Storms by Country")

    st.altair_chart(chart)

    ###############################################################

    data = pd.read_csv('./data/Storm.csv')

    total_occurrences = data["Total"].sum()

    year_columns = [str(year) for year in range(2001, 2022)]
    percentages = [data[year].sum() / total_occurrences * 100 for year in year_columns]

    df = pd.DataFrame({"Year": year_columns, "Percentage": percentages})

    st.write(f"### Contribution of Each Year's Storm Occurrences to the Total Number of Storms")

    fig = px.pie(df, values="Percentage", names="Year")

    st.plotly_chart(fig)
    

# The program loads a dataset on wildfire occurrences and displays charts showing how frequently they occur and where they occur geographically. The code also shows the percentage of total wildfires by nation and the percentage contribution of each year to the overall number of occurrences. Users can pick countries and view the wildfire counts by year. Each form of natural disaster (flood, landslide, and storm) has a unique set of graphics, and the code is repeated for each.

    
def page_seventh():
    

    df = pd.read_csv("./data/Wildfire.csv")

    countries = df['Country'].unique()

    st.write("# Wildfire Frequency by Country")

    selected_countries = st.multiselect("Select countries", countries, default=["United States", "India"])

    country_data = df[df['Country'].isin(selected_countries)].drop(['Total','ObjectId'], axis=1)

    melted_data = pd.melt(country_data, id_vars=['Country', 'Indicator'], var_name='Year', value_name='Wildfire Frequency')

    chart = alt.Chart(melted_data[melted_data['Indicator'] == 'Wildfire']).mark_bar().encode(
        x=alt.X('Year:N', title='Year', axis=alt.Axis(labelAngle=-45)),
        y=alt.Y('Wildfire Frequency:Q', title='Wildfire Frequency'),
        color=alt.Color('Country:N', legend=alt.Legend(title="Country")),
    ).properties(
        width=800,
        height=500,
        title="Wildfire Frequency by Country"
    )

    st.altair_chart(chart)

    ###############################################################

    st.write(f"### Geographical Distribution of Wildfire Occurrences Among Various Countries")
    fig = px.choropleth(data_frame=df,
                    locations='Country',
                    locationmode='country names',
                    color='Total',
                    scope='world')

    st.plotly_chart(fig)

    ###############################################################

    st.write(f"## Wildfire Count by Year for a Specific Country")

    selected_country = st.selectbox("Select a Country", countries, key='chart2')

    country_data = df[df['Country'] == selected_country].drop(['Total', 'Indicator', 'ObjectId'], axis=1)

    melted_data = pd.melt(country_data, id_vars=['Country'], var_name='Year', value_name='Wildfire_Count')

    chart2 = alt.Chart(melted_data).mark_bar(color='orange').encode(
        x=alt.X('Year:N', title='Year'),
        y=alt.Y('Wildfire_Count:Q', title='Wildfire Count'),
        tooltip=['Year', 'Wildfire_Count']
    ).properties(
        width=600,
        height=500,
        title=f"Country selected - {selected_country}"
    )

    st.altair_chart(chart2)

    ###############################################################

    wildfire_data = df[df['Indicator'] == 'Wildfire'].groupby(['Country']).sum().reset_index()

    chart = alt.Chart(wildfire_data).mark_circle().encode(
        x=alt.X('Country:N', sort='-y'),
        y=alt.Y('Total:Q', title='Total Number of Wildfires'),
        color=alt.Color('Country:N', legend=None),
        size=alt.Size('Total:Q', legend=None),
        tooltip=['Country', 'Total']
    ).properties(
        width=700,
        height=500
    ).interactive()

    st.write(f"### Proportion of Total Number of Wildfires by Country")

    st.altair_chart(chart)

    ###############################################################

    data = pd.read_csv('./data/Wildfire.csv')

    total_occurrences = data["Total"].sum()

    year_columns = [str(year) for year in range(2001, 2022)]
    percentages = [data[year].sum() / total_occurrences * 100 for year in year_columns]

    df = pd.DataFrame({"Year": year_columns, "Percentage": percentages})

    st.write(f"### Contribution of Each Year's Wildfire Occurrences to the Total Number of Wildfires")

    fig = px.pie(df, values="Percentage", names="Year")

    st.plotly_chart(fig)

    
def main():
    
    st.set_page_config(page_title="Disaster Data Hub")
    st.sidebar.title("Navigation")
    
    pages = {
        
        "Disaster Analytics": page_all_disasters,
        "Future Prediction" : prediction,
        "Drought Analysis": page_second,
        "Extreme Temperature Analysis": page_third,
        "Flood Analysis": page_fourth,
        "Landslide Analysis": page_fifth,
        "Storm Analysis": page_sixth,
        "Wildfire Analysis": page_seventh,
 
    }
    
    page = st.sidebar.selectbox("Main Menu", tuple(pages.keys()))
    pages[page]()

if __name__ == "__main__":
    main()
