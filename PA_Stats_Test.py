import pandas as pd
import numpy as np
import streamlit as st
PA_df= pd.read_csv('/Users/johndavenport/Python Projects/Pandas Test/PA_Test.csv', encoding="ISO-8859-1")
print(PA_df.to_string())
st.dataframe(PA_df)
st.write(PA_df)
PA_Filter_df= PA_df.filter

#print(PA_Filter_df.to_string())






Max_Velocity_Chart_df = pd.DataFrame({
    "Category": Players_List, 
    "Value": Max_Velocity_List, 
    "Meters Covered": Meteres_Covered_List
})

# Create bar chart
fig = px.bar(
    Max_Velocity_Chart_df, 
    x="Category", 
    y="Value", 
    color="Meters Covered",
    color_continuous_scale="viridis",  # You can change the color scale
    title="Max Velocity Chart",
    labels={"Value": "Max Velocity", "Category": "Players"}
)

# Show the figure
fig.show()