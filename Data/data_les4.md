# Data les 4 - streamlit dashboard

In deze les ga een dashboard maken waarin je de statistieken en visualisaties die je in de vorige les in een notebook gemaakt hebt, gaat weergeven. Hiervoor ga je gebruik maken van streamlit. 

## Voorbereiding voor deze les
Als het goed is, is streamlit al geinstalleerd in je conda environment ai-s3. Dit kun je controleren door een terminal (of anaconda prompt) te openen, de environment te activeren (`conda activate ai-s3`) en `streamlit hello` te runnen. Nu moet tabblad in de browser openen met een streamlit dashboard. Is dat niet het geval, dan moet je mogelijk streamlit nog installeren. Hier zijn de instructies: https://docs.streamlit.io/get-started/installation

Volg daarna deze tutorial om een eerste dashboard te maken: https://docs.streamlit.io/get-started/tutorials/create-an-app

Om interactieve visualisaties te maken kun je ook gebruik maken van plotly express, hiermee kun je met heel weinig code interactieve visualisaties krijgen. kijk hier voor voorbeelden: Om interactieve visualisaties te maken kun je gebruik maken van plotly, kijk hier voor voorbeelden: https://plotly.com/python/plotly-express/#gallery en lees hier over de combi van plotly en streamlit: https://docs.streamlit.io/develop/api-reference/charts/st.plotly_chart en lees hier over de combi van plotly en streamlit: https://docs.streamlit.io/develop/api-reference/charts/st.plotly_chart

In streamlit_demo.py wordt ook een voorbeeld getoond van een visualisatie gemaakt met seaborn en streamlit widgets en een met plotly express. Die tweede is niet per se mooier, maar wel veel makkelijker te coderen. Bekijk de code en run het dashboard met `streamlit run streamlit_demo.py` (ga hiervoor eerst in een terminal/anaconda prompt naar de juiste folder en zorg dat de conda omgeving ai-s3 is geactiveerd `conda activate ai-s3`). 

## In de les
In de les heb je tijd om je dashboard te maken. Je kunt daarbij gebruik maken van de visualisaties die je in de vorige les gemaakt hebt, je kunt nieuwe visualisaties toevoegen en deze uitbreiden met streamlit widgets en/of visualisaties met plotly om ze interactief te maken. 

Hier nog eens de instructies:

Je gaat een dashboard maken om klanten van airbnb een goed aanbod te laten zien van wat er in de door jou gekozen stad te huur is. Het is natuurlijk leuk een om kaartje te laten zien met daarop de locaties van de apparatementen, maar toon bijvoorbeeld ook een overzicht van de gemiddelde prijs per buurt, of de grootte van het aanbod per type woning. 

Bedenk steeds eerst wat je wilt laten zien en waarom, leg dit voor aan een docent of mede-student en ga dan pas de visualisaties maken. Bij het maken van de visualisaties mag je gebruik maken van een LLM, je kunt deze vragen om gebruik te maken van het *seaborn* package of van *plotly*. 


## Portfolio-item
Lever de code van het dashboard in als portfolio-item. Geef daarbij ook instructies voor het runnen van het dashboard.

Lever ook het notebook in waarin je de data hebt bewerkt en opgeschoond voordat je aan de visualisaties begon. Zorg dat notebook netjes is, een logisch geheel is en dat je de gemaakte stappen toelicht. 

### Vereisten aan het dashboard
- Maak minstens 5 visualisaties, waarvan zeker 2 interactieve.
- Zorg dat er tekst aanwezig is zodat de gebruiker weet hoe het dashboard gebruikt moet worden en wat er te zien is. Maar houd dit beknopt, hoe intuitiever hoe beter.
- Zorg dat het dashboard een logisch geheel is. 



