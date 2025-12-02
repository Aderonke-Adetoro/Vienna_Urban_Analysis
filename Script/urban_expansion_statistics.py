#read dataframe
change_df = pd.read_csv(output_dir/ "Change_Stats_2016_2025.csv") 
change_df.head()
### FILTERING CHANGE DATAFRAME FOR ONLY URBAN EXPANSION STATISTICS

# Filter: Where did the land become "Settlement"?
urban_expansion = change_df[ (change_df["To_Class"] == "settlement") & (change_df["From_Class"] != "settlement") ]

# Calculate total new settlement area
total_growth_km2 = urban_expansion["Area_km2"].sum()
total_growth_ha = urban_expansion["Area_Hectares"].sum()

print("--- URBAN EXPANSION STATISTICS ---")
print(f"Total New Settlement Area: {total_growth_km2:.2f} km²")
print(f"Total New Settlement Area: {total_growth_ha:.2f} Hectares")
print("\nBreakdown of what was lost:")
print(urban_expansion[["From_Class", "Area_km2", "Area_Hectares"]])

# Filter: The Seasonal Noise (Agriculture -> Terrain)
seasonal_change = change_df[ (change_df["From_Class"] == "vegetation") & (change_df["To_Class"] == "terrain") ]
print("\n--- SEASONAL / PHENOLOGICAL CHANGE (Likely Harvest) ---")
print(seasonal_change[["From_Class", "To_Class", "Area_km2"]])