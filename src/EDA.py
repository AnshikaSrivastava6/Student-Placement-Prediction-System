import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import DataPath
from config import Cleaned_DataPath

df=pd.read_csv(DataPath)
print(df.shape)
print(df.head())

print(df['PlacementStatus'].value_counts())
sns.countplot(x="PlacementStatus",data=df)
plt.savefig("images/placement_count.png",dpi=300,bbox_inches='tight')


#categorical analysis

features=['ExtracurricularActivities','PlacementTraining']
colors=['skyblue','salmon']
fig,axes=plt.subplots(1,2 ,figsize=(14,10))
axes=axes.flatten()
for i,col in enumerate(features):
    sns.countplot(x=col,hue='PlacementStatus',data=df,ax=axes[i],color=colors[i])
    axes[i].set_title(f'{col} vs PlacementStatus')
plt.tight_layout()   
plt.savefig("images/categorical_analysis.png",dpi=300,bbox_inches='tight')
plt.show()

#numerical analysis
features_1=['CGPA','Internships','Projects','Workshops/Certifications']
colors=['skyblue','salmon','green','blue']
fig,axes=plt.subplots(2,2,figsize=(14,10))
axes=axes.flatten()
for i,col in enumerate(features_1):
    sns.boxplot(x='PlacementStatus',y=col,data=df,ax=axes[i],color=colors[i])
    axes[i].set_title(f'{col} vs Placement_status')
plt.savefig("images/numerical_analysis.png",dpi=300,bbox_inches='tight')
plt.tight_layout()   

features_1=['SSC_Marks','HSC_Marks','AptitudeTestScore','SoftSkillsRating']
colors=['skyblue','salmon','green','blue']
fig,axes=plt.subplots(2,2,figsize=(14,10))
axes=axes.flatten()
for i,col in enumerate(features_1):
    sns.boxplot(x='PlacementStatus',y=col,data=df,ax=axes[i],color=colors[i])
    axes[i].set_title(f'{col} vs Placement_status')
plt.savefig("images/skill_analysis.png",dpi=300,bbox_inches='tight')    
plt.tight_layout()   


#Distribution analysis
numerical_cols = [
    'CGPA',
    'SSC_Marks',
    'HSC_Marks',
    'AptitudeTestScore',
    'SoftSkillsRating'
]
colors=['skyblue','salmon','green','blue','violet']
fig,axes=plt.subplots(1,5,figsize=(14,10))
axes=axes.flatten()
for i,col in enumerate(numerical_cols):
    sns.histplot(x=df[col],kde=True,ax=axes[i],color=colors[i])
    axes[i].set_title(f'{col} distribution')
plt.savefig("images/distribution_analysis.png",dpi=300,bbox_inches='tight')    
plt.tight_layout()   


count_cols = [
    'Internships',
    'Projects',
    'Workshops/Certifications'
]
colors=['skyblue','salmon','green']
fig,axes=plt.subplots(1,3,figsize=(14,10))
axes=axes.flatten()
for i,col in enumerate(count_cols):
    sns.countplot(x=df[col],ax=axes[i],color=colors[i])
    axes[i].set_title(f'{col} count')
plt.savefig("images/count_feature.png",dpi=300,bbox_inches='tight')    
plt.tight_layout()   


#Correlation heatmap
df_cleaned=pd.read_csv(Cleaned_DataPath)
plt.figure(figsize=(20,14))
corr=df_cleaned.corr(numeric_only=True)
sns.heatmap(corr,annot=True,cmap='coolwarm')
plt.savefig("images/heatmap.png",dpi=300,bbox_inches='tight')



