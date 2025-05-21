import streamlit as st #UI/UX
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler # scale data
from sklearn.metrics import pairwise_distances, mean_squared_error #import pairwise distances for manual KNN calculation, built in KNN model is not as malleable, MSE for evaluation
import warnings
import random # For sampling

# Streamlit Config and Warnings (clear up terminal)
st.set_page_config(layout="wide", page_title="Plant Recommender", page_icon="🌱")
warnings.filterwarnings('ignore', category=UserWarning, message='X does not have valid feature names')
warnings.filterwarnings('ignore', category=FutureWarning)

# Constants
NUM_RECOMMENDATIONS_TO_SHOW = 5
METRIC_FOR_KNN = 'cosine'
NO_PREFERENCE_STR = "No Preference / Any"

# Load and Preprocess Data
@st.cache_data #loads and stores results of the pre-processing once. improves app speed on further executions.
def load_and_preprocess_data():
    try:
        df = pd.read_csv('usda_cleaned_plant_data.csv')
    except FileNotFoundError:
        st.error("Error: usda_cleaned_plant_data.csv not found.")
        return None, None, None, None, None, None, None, None, None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None, None, None, None, None, None

    df_processed = df.copy()
    df_processed['pH_Minimum'].fillna(7.0, inplace=True)
    df_processed['pH_Maximum'].fillna(7.0, inplace=True)
    df_processed['Temperature_Minimum_F'].fillna(0, inplace=True)
    ordinal_features_impute = ['Growth_Rate', 'Lifespan', 'Toxicity', 'Drought_Tolerance',
                               'Hedge_Tolerance', 'Moisture_Use', 'Salinity_Tolerance',
                               'Shade_Tolerance']
    
    for col in ordinal_features_impute:
        if col in df_processed.columns and df_processed[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df_processed[col]):
                median_val = df_processed[col].median()
                df_processed[col].fillna(median_val, inplace=True)

    all_cols_except_id_name = [col for col in df_processed.columns if col not in ['id', 'Scientific_Name_x']]
    for col in all_cols_except_id_name:
        if df_processed[col].isnull().any():
            if col in df_processed.columns and pd.api.types.is_numeric_dtype(df_processed[col]):
                if df_processed[col].nunique() < 10 or df_processed[col].dtype == 'float64':
                     df_processed[col].fillna(df_processed[col].median(), inplace=True)
                else:
                     df_processed[col].fillna(0, inplace=True)
            else:
                df_processed[col].fillna(0, inplace=True)

    identifiers = ['id', 'Scientific_Name_x']
    feature_cols = [col for col in df_processed.columns if col not in identifiers]
    features_df = df_processed[feature_cols].copy()
    for col in features_df.columns:
        if not pd.api.types.is_numeric_dtype(features_df[col]):
            try: features_df[col] = pd.to_numeric(features_df[col])
            except ValueError: features_df[col] = 0

    default_values = {}
    for col in features_df.columns:
        is_one_hot_like = features_df[col].isin([0, 1]).all() and \
                          col not in ordinal_features_impute and \
                          col not in ['pH_Minimum', 'pH_Maximum', 'Temperature_Minimum_F']
        if is_one_hot_like: default_values[col] = 0.0
        elif features_df[col].nunique() < 10 and features_df[col].min() >=0 : default_values[col] = features_df[col].median()
        else: default_values[col] = features_df[col].mean()

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_df.values)

    one_hot_groups = {
        'Category': [c for c in df.columns if c.startswith('Category_')],
        'Family': [c for c in df.columns if c.startswith('Family_')],
        'Growth_Habit': [c for c in df.columns if c.startswith('Growth_Habit_')],
        'Native_Status': [c for c in df.columns if c.startswith('Native_Status_')],
        'Active_Growth_Period': [c for c in df.columns if c.startswith('Active_Growth_Period_')],
        'Fall_Conspicuous': [c for c in df.columns if c.startswith('Fall_Conspicuous_')],
        'Flower_Color': [c for c in df.columns if c.startswith('Flower_Color_')],
        'Flower_Conspicuous': [c for c in df.columns if c.startswith('Flower_Conspicuous_')],
        'Fruit_Conspicuous': [c for c in df.columns if c.startswith('Fruit_Conspicuous_')],
        'Bloom_Period': [c for c in df.columns if c.startswith('Bloom_Period_')],
        'Fire_Resistance': [c for c in df.columns if c.startswith('Fire_Resistance_')]
    }
    return df, df_processed, features_df, features_scaled, scaler, default_values, one_hot_groups, feature_cols, ordinal_features_impute

df_original, df_processed_global, features_df_global, features_scaled_global, scaler_global, default_values_global, one_hot_groups_global, feature_cols_global, ordinal_features_impute_global = load_and_preprocess_data()


# Recommendation Function (Manual KNN)
def recommend_plants_knn_ideal_profile(criteria, k=10, metric='cosine',
                                       current_plant_name_to_exclude=None): # exclusion param for evaluation
    if df_original is None: return pd.DataFrame()
    query_vector = default_values_global.copy()

    for key, value in criteria.items():
        if value is None: continue
        if key in query_vector and key not in one_hot_groups_global:
            if key == 'pH_Minimum':
                if 'pH_Maximum' in criteria and criteria['pH_Maximum'] is not None:
                    try:
                        min_val = float(value); max_val = float(criteria['pH_Maximum'])
                        mid_ph = (min_val + max_val) / 2
                        query_vector['pH_Minimum'] = mid_ph
                        query_vector['pH_Maximum'] = mid_ph
                    except ValueError: pass
                else:
                    try: query_vector[key] = float(value)
                    except ValueError: pass
            elif key == 'pH_Maximum':
                if 'pH_Minimum' not in criteria or criteria.get('pH_Minimum') is None:
                    try: query_vector[key] = float(value)
                    except ValueError: pass
            else:
                try: query_vector[key] = float(value)
                except (ValueError, TypeError): pass
        elif key in one_hot_groups_global:
            desired_one_hot_cols_for_group = value
            for col_in_group in one_hot_groups_global[key]:
                if col_in_group in query_vector: query_vector[col_in_group] = 0.0
            if desired_one_hot_cols_for_group:
                for desired_col in desired_one_hot_cols_for_group:
                    if desired_col in query_vector:
                        query_vector[desired_col] = 1.0
        # else: st.warning(f"Criterion key '{key}' not found. Ignoring for query.")

    query_df = pd.DataFrame([query_vector], columns=feature_cols_global)
    for col in feature_cols_global:
        if col not in query_df.columns: query_df[col] = default_values_global.get(col, 0)
    query_df = query_df[feature_cols_global]
    for col in query_df.columns:
        if query_df[col].isnull().any(): query_df[col].fillna(default_values_global.get(col, 0), inplace=True)
    query_unscaled = query_df.values

    try:
        query_scaled = scaler_global.transform(query_unscaled)
        if np.isnan(query_scaled).any() or np.isinf(query_scaled).any():
             st.error("Error: Scaled query vector NaN/Inf."); return pd.DataFrame()
    except Exception as e:
        st.error(f"Error scaling query: {e}"); return pd.DataFrame()

    try:
        if query_scaled.shape[1] != features_scaled_global.shape[1]:
            st.error(f"Shape mismatch! QF: {query_scaled.shape[1]}, DF: {features_scaled_global.shape[1]}"); return pd.DataFrame()
        distances = pairwise_distances(query_scaled, features_scaled_global, metric=metric)
        distances_flat = distances.flatten()

        # Sort by distance
        sorted_indices_by_distance = np.argsort(distances_flat)

        # Exclude the current_plant_name_to_exclude if provided
        recommendation_indices = []
        for idx in sorted_indices_by_distance:
            if current_plant_name_to_exclude and df_original.iloc[idx]['Scientific_Name_x'] == current_plant_name_to_exclude:
                continue
            recommendation_indices.append(idx)
            if len(recommendation_indices) >= k:
                break
        # st.write(f"Found {len(recommendation_indices)} closest candidates.") # Debug
    except Exception as e:
        st.error(f"Error in distance calc/sort: {e}"); return pd.DataFrame()

    return df_original.iloc[recommendation_indices]


#UI Helper Functions
def get_float_input_st(prompt_label, default_value=None, allow_blank=True):
    val = st.sidebar.number_input(
        prompt_label, value=default_value if default_value is not None else np.nan,
        step=1.0, format="%.1f" if "pH" in prompt_label else "%.0f",
        key=prompt_label.replace(" ", "_").replace("(", "").replace(")", "").replace("°", "")
    )
    return None if np.isnan(val) else val

def get_ordinal_selectbox_st(prompt_label, options_map_num_to_display, default_display_option=NO_PREFERENCE_STR):
    display_options = [NO_PREFERENCE_STR] + list(options_map_num_to_display.values())
    selected_display_str = st.sidebar.selectbox(prompt_label, options=display_options, index=0)
    if selected_display_str == NO_PREFERENCE_STR: return None
    for num_val, display_str in options_map_num_to_display.items():
        if display_str == selected_display_str: return num_val
    return None

def get_multiselect_options(group_key_in_one_hot_groups):
    options_for_multiselect = {}
    cols = one_hot_groups_global.get(group_key_in_one_hot_groups, []) # Use global
    valid_cols = [col for col in cols if not col.endswith("_nan")]
    for col_name in valid_cols:
        display_name = col_name.replace(group_key_in_one_hot_groups + "_", "").replace("_", " ")
        display_name = display_name.replace("/herb,", "/herb, ")
        options_for_multiselect[display_name] = col_name
    return options_for_multiselect

# Streamlit Web App Design and Functionality
st.title("🌱 Plant Recommender 🌱")
st.markdown("Select your desired plant features in the sidebar to get recommendations.")

if df_original is None:
    st.info("Data could not be loaded. Please check the CSV file and console output.")
    st.stop()

st.sidebar.header("Plant Criteria")
user_criteria_streamlit = {}

temp_min_user = get_float_input_st(
    "Max tolerable LOWEST Temperature (°F) (e.g., -20)", default_value=None
)
if temp_min_user is not None: user_criteria_streamlit['Temperature_Minimum_F'] = temp_min_user

st.sidebar.subheader("Soil pH Range")
ph_min_user = get_float_input_st("Desired MINIMUM soil pH (e.g., 5.0)", default_value=None)
ph_max_user = get_float_input_st("Desired MAXIMUM soil pH (e.g., 7.0)", default_value=None)
if ph_min_user is not None: user_criteria_streamlit['pH_Minimum'] = ph_min_user
if ph_max_user is not None: user_criteria_streamlit['pH_Maximum'] = ph_max_user

st.sidebar.subheader("Tolerances & Characteristics")
ordinal_feature_ui_map = {
    'Drought_Tolerance': ("Drought Tolerance", {1:"Low", 2:"Medium", 3:"High"}),
    'Shade_Tolerance': ("Shade Tolerance", {0:"Full Sun", 1:"Partial", 2:"Full Shade"}),
    'Salinity_Tolerance': ("Salinity Tolerance", {0:"None", 1:"Low", 2:"Medium", 3:"High"}),
    'Growth_Rate': ("Growth Rate", {1:"Slow", 2:"Moderate", 3:"Rapid"}),
    'Lifespan': ("Lifespan", {1:"Short", 2:"Medium", 3:"Long"}),
    'Toxicity': ("Maximum Acceptable Toxicity", {0:"None", 1:"Minor", 2:"Moderate", 3:"High/Any"})
}
for feature_key, (label, num_to_display_map) in ordinal_feature_ui_map.items():
    default_display = NO_PREFERENCE_STR
    if feature_key == 'Toxicity': default_display = num_to_display_map.get(3, NO_PREFERENCE_STR)
    selected_value = get_ordinal_selectbox_st(label, num_to_display_map, default_display_option=default_display)
    if selected_value is not None:
        if not (feature_key == 'Toxicity' and selected_value == 3):
            user_criteria_streamlit[feature_key] = selected_value

st.sidebar.subheader("Categorical Features")
gh_options_map = get_multiselect_options("Growth_Habit")
if gh_options_map:
    selected_gh_display = st.sidebar.multiselect("Preferred Growth Habit(s)", options=list(gh_options_map.keys()))
    if selected_gh_display:
        user_criteria_streamlit['Growth_Habit'] = [gh_options_map[name] for name in selected_gh_display]

bp_options_map = get_multiselect_options("Bloom_Period")
if bp_options_map:
    selected_bp_display = st.sidebar.multiselect("Preferred Bloom Period(s)", options=list(bp_options_map.keys()))
    if selected_bp_display:
        user_criteria_streamlit['Bloom_Period'] = [bp_options_map[name] for name in selected_bp_display]

if "Flower_Conspicuous_Yes" in feature_cols_global:
    flower_conspic = st.sidebar.checkbox("Showy Flowers?", value=False)
    if flower_conspic: user_criteria_streamlit['Flower_Conspicuous'] = ["Flower_Conspicuous_Yes"]

if "Fall_Conspicuous_Yes" in feature_cols_global:
    fall_conspic = st.sidebar.checkbox("Fall Conspicuous?", value=False)
    if fall_conspic: user_criteria_streamlit['Fall_Conspicuous'] = ["Fall_Conspicuous_Yes"]

if st.sidebar.button("🌿 Find Matching Plants 🌿"):
    if not user_criteria_streamlit:
        st.warning("Please select at least one criterion to get recommendations.")
    else:
        st.subheader("Your Selected Criteria for Ideal Profile:")
        for crit_key_display, crit_val in user_criteria_streamlit.items():
            display_key = crit_key_display.replace('_', ' ')
            if isinstance(crit_val, list):
                display_values = [v.split('_')[-1].replace("_", " ") for v in crit_val]
                st.write(f"- **{display_key}**: {', '.join(display_values)}")
            elif crit_key_display in ordinal_feature_ui_map:
                display_val_str = ordinal_feature_ui_map[crit_key_display][1].get(crit_val, str(crit_val))
                st.write(f"- **{display_key}**: {display_val_str}")
            else:
                st.write(f"- **{display_key}**: {crit_val}")
        st.markdown("---")


        recommended_plants_df = recommend_plants_knn_ideal_profile(
            user_criteria_streamlit,
            k=NUM_RECOMMENDATIONS_TO_SHOW,
            metric=METRIC_FOR_KNN
        )

        if not recommended_plants_df.empty:
            st.subheader(f"Top {len(recommended_plants_df)} Plant Recommendations:")
            for i, (index, row) in enumerate(recommended_plants_df.iterrows()):
                st.markdown(f"#### {i+1}. {row['Scientific_Name_x']} (ID: {row['id']})")
                with st.expander("Show All Features"):
                    st.write("Key Numeric & Ordinal Features:")
                    key_numerics_ordinals = ['Growth_Rate', 'Lifespan', 'Toxicity', 'Drought_Tolerance',
                                    'Hedge_Tolerance', 'Moisture_Use', 'pH_Minimum', 'pH_Maximum',
                                    'Salinity_Tolerance', 'Shade_Tolerance', 'Temperature_Minimum_F']
                    for kno_col in key_numerics_ordinals:
                        if kno_col in row and pd.notna(row[kno_col]):
                            display_val = row[kno_col]
                            if kno_col in ordinal_feature_ui_map:
                                try:
                                    num_val_for_map = int(float(row[kno_col]))
                                    display_val = ordinal_feature_ui_map[kno_col][1].get(num_val_for_map, str(row[kno_col]))
                                except ValueError:
                                    display_val = str(row[kno_col])
                            st.write(f"  - **{kno_col.replace('_', ' ')}**: {display_val}")

                    st.write("\nOther Categorical Features (Present):")
                    for group_key_display, cols_in_group in one_hot_groups_global.items():
                        present_features_in_group = []
                        for col_name_actual in cols_in_group:
                            if col_name_actual in row and row[col_name_actual] == 1:
                                display_feature_name = col_name_actual.replace(group_key_display + "_", "").replace("_", " ")
                                present_features_in_group.append(display_feature_name)
                        if present_features_in_group:
                            st.write(f"  - **{group_key_display.replace('_', ' ')}**: {', '.join(present_features_in_group)}")
                st.markdown("---")

        else:
            st.info("No plants found closely matching your ideal profile based on the selected criteria. Try adjusting your preferences.")
else:
    st.info("Adjust criteria in the sidebar and click 'Find Matching Plants'.")

# Footer/Citation
st.markdown("---")
st.caption("Data Source: Natural Resources Conservation Service. PLANTS Database. United States Department of Agriculture. Accessed May 10, 2024 (example date), from https://plants.usda.gov.")


# Evaluation (comment out for normal use or leave in because it doesn't really mess up the design)
st.sidebar.markdown("---")
st.sidebar.header("Model Evaluation")
num_eval_samples = st.sidebar.slider("Number of plants to evaluate against:", 1, 50, 10)

if st.sidebar.button("🧪 Evaluate KNN Recommendations"):
    if df_original is None:
        st.error("Data not loaded, cannot evaluate.")
    else:
        st.subheader("KNN Model Evaluation")
        evaluation_results = []
        # Take a random sample of plants from the original dataframe for evaluation
        sample_indices = random.sample(range(len(df_original)), min(num_eval_samples, len(df_original)))
        test_plants_df = df_original.iloc[sample_indices]

        total_rank = 0
        found_in_top_k = 0
        total_mse_dict = {feat: 0 for feat in ordinal_features_impute_global + ['Temperature_Minimum_F', 'pH_Minimum', 'pH_Maximum']}
        num_compared_features = {feat: 0 for feat in total_mse_dict}


        for idx, original_plant_row in test_plants_df.iterrows():
            original_plant_name = original_plant_row['Scientific_Name_x']
            #st.write(f"Evaluating against: **{original_plant_name}**")

            # Construct ideal criteria from the plant's actual features
            ideal_criteria = {}
            # Numerical/Ordinal
            for col in ordinal_features_impute_global + ['Temperature_Minimum_F', 'pH_Minimum', 'pH_Maximum']:
                if col in original_plant_row and pd.notna(original_plant_row[col]):
                    ideal_criteria[col] = original_plant_row[col]
            # One-hot
            for group, cols in one_hot_groups_global.items():
                active_cols = [c for c in cols if original_plant_row.get(c) == 1]
                if active_cols:
                    ideal_criteria[group] = active_cols

            # Get recommendations, excluding the plant itself from the top K
            recommendations_df = recommend_plants_knn_ideal_profile(
                ideal_criteria,
                k=10, # Get 10 neighbors for evaluation
                metric=METRIC_FOR_KNN,
                current_plant_name_to_exclude=original_plant_name
            )

            rank = -1
            if not recommendations_df.empty:
                rec_names = recommendations_df['Scientific_Name_x'].tolist()
                if original_plant_name in rec_names: # This should not happen since original plant removed from df, so check for it
                    st.warning(f"Original plant {original_plant_name} found in its own recommendations (after exclude attempt).")
                # For evaluation, we'd ideally want to see if a very similar plant is found.
                # just checking if the process runs and analyze feature similarity.

                # Feature similarity check with the top recommended plant
                if not recommendations_df.empty:
                    top_rec_row = recommendations_df.iloc[0]
                    found_in_top_k +=1 # Count this as a "successful" recommendation run for now

                    for feature_key in total_mse_dict.keys():
                        if pd.notna(original_plant_row.get(feature_key)) and pd.notna(top_rec_row.get(feature_key)):
                            diff_sq = (original_plant_row[feature_key] - top_rec_row[feature_key])**2
                            total_mse_dict[feature_key] += diff_sq
                            num_compared_features[feature_key] += 1

            evaluation_results.append({
                'Original Plant': original_plant_name,
                'Top Recommendation': recommendations_df.iloc[0]['Scientific_Name_x'] if not recommendations_df.empty else "None"
            })
            #st.write(f"  Top rec: {recommendations_df.iloc[0]['Scientific_Name_x'] if not recommendations_df.empty else 'None'}")


        st.write("--- Evaluation Summary ---")
        if evaluation_results:
            st.write(f"Evaluated {len(evaluation_results)} plants.")
            st.write(f"Number of times recommendations were generated: {found_in_top_k}")

            st.write("Average Mean Squared Error for features of top recommendation vs original plant:")
            for feat, total_err in total_mse_dict.items():
                if num_compared_features[feat] > 0:
                    avg_mse = total_err / num_compared_features[feat]
                    st.write(f"  - {feat}: {avg_mse:.2f} (based on {num_compared_features[feat]} comparisons)")
                else:
                    st.write(f"  - {feat}: Not enough data to compare.")

            st.dataframe(pd.DataFrame(evaluation_results))
        else:
            st.write("No evaluation results to display.")