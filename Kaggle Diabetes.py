#!/usr/bin/env python
# coding: utf-8

# In[67]:


pip install pandas


# In[68]:


import pandas as pd
df = pd.read_csv('Downloads/diabetes.csv')


# In[69]:


df.head()


# In[70]:


pip install pandas-profiling


# In[71]:


import pandas_profiling as pp
pp.ProfileReport(df)


# In[75]:


pip install plotly


# In[76]:


# Visualization Imports
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.express as px
import numpy as np


# In[77]:


dist = df['Outcome'].value_counts()
colors = ['mediumturquoise', 'darkorange']
trace = go.Pie(values=(np.array(dist)),labels=dist.index)
layout = go.Layout(title='Diabetes Outcome')
data = [trace]
fig = go.Figure(trace,layout)
fig.update_traces(marker=dict(colors=colors, line=dict(color='#000000', width=2)))
fig.show()


# 

# In[78]:


def df_to_plotly(df):
    return {'z': df.values.tolist(),
            'x': df.columns.tolist(),
            'y': df.index.tolist() }


# In[79]:


print(df)


# In[80]:


import plotly.graph_objects as go


# In[81]:


dfNew = df.corr()
fig = go.Figure(data=go.Heatmap(df_to_plotly(dfNew)))
fig.show()


# In[82]:


fig = px.scatter(df, x='Glucose', y='Insulin')
fig.update_traces(marker_color="turquoise",marker_line_color='rgb(8,48,107)',
                  marker_line_width=1.5)
fig.update_layout(title_text='Glucose and Insulin')
fig.show()


# In[83]:


fig = px.box(df, x='Outcome', y='Age')
fig.update_traces(marker_color="midnightblue",marker_line_color='rgb(8,48,107)',
                  marker_line_width=1.5)
fig.update_layout(title_text='Age and Outcome')
fig.show()


# In[84]:


plot = sns.boxplot(x='Outcome',y="BMI",data=df)


# In[ ]:




