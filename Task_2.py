# -------------------------------------------------------------------
# TASK 2: UNEMPLOYMENT ANALYSIS WITH PYTHON
# -------------------------------------------------------------------

# Step 1: Import libraries and load data
import pandas as pd
import plotly.express as px

df = pd.read_csv('Unemployment_Rate_upto_11_2020.csv')

# Step 2: Data Cleaning and Preprocessing
df.columns = ['Region', 'Date', 'Frequency', 'Estimated Unemployment Rate', 
              'Estimated Employed', 'Estimated Labour Participation Rate', 
              'Region.1', 'longitude', 'latitude']

df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df['Month'] = df['Date'].dt.month_name()
df['Year'] = df['Date'].dt.year

print("--- Data Cleaned and Ready ---")
print(df.head())

# Step 3: Exploratory Data Analysis & Visualization

# Plot 1: National Unemployment Trend
print("\n--- Generating Plot 1: National Unemployment Trend ---")
national_avg_unemployment = df.groupby(['Date'])['Estimated Unemployment Rate'].mean().reset_index()
fig1 = px.line(national_avg_unemployment, x='Date', y='Estimated Unemployment Rate', 
               title='National Unemployment Rate Trend in India')
fig1.show()

# Plot 2: State-wise Unemployment during Lockdown Peak (April 2020)
print("--- Generating Plot 2: State-wise Unemployment during Lockdown ---")
lockdown_data = df[df['Date'].dt.strftime('%Y-%m') == '2020-04']
fig2 = px.bar(lockdown_data.sort_values('Estimated Unemployment Rate', ascending=False), 
              x='Region', y='Estimated Unemployment Rate',
              color='Estimated Unemployment Rate',
              title='Unemployment Rate by State during Covid-19 Lockdown (April 2020)',
              labels={'Region': 'State', 'Estimated Unemployment Rate': 'Unemployment Rate (%)'})
fig2.update_layout(xaxis={'categoryorder':'total descending'})
fig2.show()

# Plot 3: Interactive map of unemployment over time
print("--- Generating Plot 3: Interactive Map of Unemployment ---")
fig3 = px.scatter_geo(df, 'longitude', 'latitude', color="Estimated Unemployment Rate",
                      hover_name="Region", size="Estimated Unemployment Rate",
                      animation_frame="Month", scope='asia', 
                      title='State-wise Unemployment Rate Over Time')
fig3.update_geos(lataxis_range=[5, 35], lonaxis_range=[65, 100], oceancolor="#6dd5ed",
                 showocean=True)
fig3.show()

# Step 4 & 5: Insights and Policy Recommendations (as detailed in the text above)
print("\n--- Analysis Complete ---")
print("Key Insights:")
print("1. Massive unemployment spike to >23% during the April-May 2020 lockdown.")
print("2. States like Jharkhand, Bihar, and Haryana were disproportionately affected.")
print("3. A gradual but uneven recovery followed the easing of restrictions.")

print("\nPolicy Recommendations:")
print("1. Strengthen social safety nets for informal sector workers.")
print("2. Deploy targeted economic relief to the hardest-hit states.")
print("3. Promote resilient employment sectors and invest in digital skill development.")