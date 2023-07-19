# Importación de las librerías
import pandas as pd
import pickle
import streamlit as st
from datetime import datetime,timedelta
from sklearn.preprocessing import StandardScaler
import folium
from streamlit_folium import folium_static
import numpy as np
# Librerías SMS
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse

# Carga del modelo de ML
model=pickle.load(open("model_ml.pickle","rb"))

# Lista de estados de E.E.U.U y sus abreviaturas
state = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'North Carolina', 'South Carolina', 'Colorado',
           'Connecticut', 'North Dakota', 'South Dakota', 'Delaware', 'Florida', 'Hawaii', 'Idaho', 'Illinois',
           'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan',
           'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey',
           'New York', 'New Mexico', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'Tennessee', 'Texas',
           'Utah', 'Vermont', 'Virginia', 'West Virginia', 'Washington', 'Wisconsin', 'Wyoming', 'District of Columbia']

abbreviation = ['AL', 'AK', 'AZ', 'AR', 'CA', 'NC', 'SC', 'CO', 'CT', 'ND', 'SD', 'DE', 'FL', 'GA', 'HI', 'ID', 'IL',
                'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
                'NY', 'NM', 'OH', 'OK', 'OR', 'PA', 'RI', 'TN', 'TX', 'UT', 'VT', 'VA', 'WV', 'WA', 'WI', 'WY', 'DC']

# Función para la estandarización de la magnitud
def apply_filter(df):
    filter = (df.magType == "ml") & (df.mag >= 3.3) & (df.mag <= 6.6)
    df["magType"] = df.magType.mask(filter, "mw")
    df["mag"] = df.mag.mask(filter, 0.8095 * df.mag + 1.3003)

    df["magType"] = df.magType.mask(df.magType.isin(["mww", "mwc", "mwb", "mwr"]), "mw")

    def filter2(mb):
        ms = 1.664 * mb - 3.753 if mb <= 5.9 else 2.763 * mb - 10.301
        if 3 < ms < 6.1:
            return 0.67 * ms + 2.07, True
        elif 6.2 < ms < 8.2:
            return 0.99 * ms + 0.08, True
        else:
            return mb, False
    mask = df["mag"].apply(filter2).apply(lambda x: x[1])
    df.loc[mask, "magType"] = "mw"
    df.loc[mask, "mag"] = df.loc[mask, "mag"].apply(filter2).apply(lambda x: x[0])
    
    df["depth"]=df.depth.round().astype(int)
    df["mag"]=df.mag.round(1)

    return df[df.magType=="mw"]

# Carga de datos de la API (últimos 30 días) y ETL
start_date=pd.to_datetime(datetime.now())-timedelta(days=30)
end_date = pd.to_datetime(datetime.now())+timedelta(days=1)
url = f"https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv&starttime={start_date.strftime('%Y-%m-%d')}&endtime={end_date.strftime('%Y-%m-%d')}"
df=pd.read_csv(url)
df.drop_duplicates(keep=False,inplace=True)
df.dropna(subset=["latitude","longitude","depth","mag","magType","place"],inplace=True)
df=df[df.type=="earthquake"]
df["country_state"]=df.place.apply(lambda x: x.split(", ")[-1])
df= df[
    (df.country_state.apply(lambda x: x in state or x in abbreviation)) |
    df.country_state.str.contains("Japan") |
    df.country_state.str.contains("Peru")
]
df["country"] = df.country_state.apply(lambda x: "USA" if ("Peru" not in x) and ("Japan" not in x) else ("Japan" if "Japan" in x else "Peru"))
df=apply_filter(df)
df["time"]=pd.to_datetime(df.time)
df=df[["time","latitude","longitude","depth","mag","place","country"]].sort_values(by="time",ascending=False)

# Predicción
df_pred=df.drop(columns=["time","place","country"])
ss=StandardScaler()
df_pred=ss.fit_transform(df_pred)
y_pred=model.predict(df_pred)

# Añadiendo la columna de peligrosidad
df["hazard"]=y_pred
df["hazard"]=df.hazard.apply(lambda x: "High" if x==0 else ("Medium" if x==1 else "Low"))

### SMS
# Dataframe SMS
df_sms=df[df.country=="Peru"].sort_values(by="time",ascending=False)

# Credenciales SMS
account_sid = 'AC7b251f302c1219ff41ce9cfecf982895'
auth_token = '5cb0981ddb5b22043439d5c6fec9e274'
twilio_whatsapp_number = 'whatsapp:+14155238886'
twilio_sms_number = '+12056228055'  
recipient_phone_number = '+573229418057'
recipient_whatsapp_number = 'whatsapp:+573229418057'

# Funciones SMS
def send_whatsapp(message_body):
    client = Client(account_sid, auth_token)
    message = client.messages.create(
        body=message_body,
        from_=twilio_whatsapp_number,
        to=recipient_whatsapp_number
    )
    return message.sid

def send_sms(message_body):
    client = Client(account_sid, auth_token)
    message = client.messages.create(
        body=message_body,
        from_=twilio_sms_number,
        to=recipient_phone_number
    )
    return message.sid

### Visualizaciones

# Título
st.title("Earthquake Map - Last 30 Days")

# Botones de país
selected_country=st.sidebar.radio("Country",["All", "USA", "Japan", "Peru"],index=0)

# Botones de gravedad del sismo
selected_hazard=st.sidebar.radio("Earthquake Severity",["All", "High", "Medium", "Low"],index=0)

# Desplegable de fechas  
array=df.time.dt.strftime("%Y-%m-%d").unique()
array=np.insert(array,0,"All")
selected_date=st.selectbox("Select a date",array)
if selected_date!="All":
    df=df[df.time.dt.strftime("%Y-%m-%d")==selected_date]

# Filtros del DataFrame según los botones seleccionados
if selected_country=="All" and selected_hazard == "All":
    filtered_df=df
elif selected_country=="All" and selected_hazard!="All":
    filtered_df=df[df.hazard==selected_hazard]
elif selected_country!="All" and selected_hazard=="All":
    filtered_df=df[df.country==selected_country]
else:
    filtered_df=df[(df.country==selected_country)&(df.hazard==selected_hazard)]

# Verificación si filtered_df está vacío
if filtered_df.empty:
    m=folium.Map(location=[0,0],zoom_start=1)
    st.subheader("Map")
    folium_static(m)

    df=df[["time","mag","depth","country","hazard","place"]]
    df=df.rename(columns={"time":"date(UTC)","mag":"magnitude(Mw)","depth":"depth(km)"})
    empty_df=pd.DataFrame(columns=df.columns)
    st.subheader("Table")
    st.dataframe(empty_df)
else:
    # Creación del mapa filtrado
    m=folium.Map(location=[filtered_df["latitude"].mean(), filtered_df["longitude"].mean()], zoom_start=1)
    for index, row in filtered_df.iterrows():
        color = "red" if row["hazard"] == "High" else "yellow" if row["hazard"] == "Medium" else "green"
        folium.Circle(
            location=[row["latitude"], row["longitude"]],
            radius=500,
            color=color,
            fill=True,
            fill_color=color
        ).add_to(m)

    # Visualización del mapa en la página
    st.subheader("Map")
    folium_static(m)

    # Visualización del DataFrame filtrado
    st.subheader("Table")
    filtered_df=filtered_df[["time","mag","depth","country","hazard","place"]]
    filtered_df=filtered_df.rename(columns={"time":"date(UTC)","mag":"magnitude(Mw)","depth":"depth(km)"})
    st.dataframe(filtered_df.reset_index(drop=True))

# Creación del botón que redigirá a la página del Dashboard
if st.button("Link Earthquake Dashboard"):
    url="https://app.powerbi.com/view?r=eyJrIjoiODI0MWI3ODEtZmM2My00NGM1LWI5YmYtNWRmMGFkYWU4ZjNjIiwidCI6IjJiZTM1NDkyLWEzMmEtNDBiNS1hOWY4LWZmMjMxMTBmZDBhYyIsImMiOjR9&pageName=ReportSectiona4a149bcb98fb302c702"
    st.markdown(f"[Go to Earthquake Dashboard]({url})")

# Imagen relacionada con el botón del Dashboard
image_url="Dashboard.png"
st.image(image_url)

# Creación del botón que envía el mensaje de sismo
if st.button("Alert Test Message"):
    latest_earthquake_data = df_sms.iloc[0]
    fecha_utc = latest_earthquake_data['time']
    place = latest_earthquake_data['place']
    profundidad = latest_earthquake_data['depth']
    magnitud = latest_earthquake_data['mag']
    hazard = latest_earthquake_data['hazard']

    message_body = f"SAP - Seismic Alert Pro.\n¿Estas bien? Hemos registrado un sismo en Perú\nÚltimo sismo en Perú:\nFecha UTC: {fecha_utc} \nMagnitud (Mw): {magnitud} \nPeligrosidad: {hazard} \nEpicentro: {place} \nProfundidad (km): {profundidad}"

    whatsapp_message_id = send_whatsapp(message_body)
    sms_message_id = send_sms(message_body)

    print("WhatsApp enviado correctamente. ID del mensaje:", whatsapp_message_id)
    print("SMS enviado correctamente. ID del mensaje:", sms_message_id)