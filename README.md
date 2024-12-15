# Federated Learning

Joshua Hall en Jurrean De Nys

Dit project gebruikt federated learning om een model te trainen. 

Om dit project te runnen moet je de main file runnen. 

## Inleiding

Deze implementatie demonstreert een eenvoudig Federated Learning-model voor spamdetectie met behulp van `SGDClassifier` uit scikit-learn. Het model wordt getraind op een spamdataset en maakt gebruik van np.mean om het gemiddelde modelgewichten van de clients te krijgen.

## Datasplitsing

1. **Dataset laden en preprocessen:**
    - De dataset wordt geladen uit `data/spam.csv`.
    - De labels worden gemapt naar `spam` = 1, `ham` = 0.

2. **Splitsen van de data:**
    - De data wordt gesplitst in een trainingsset (80%) en een testset (20%).
    - De trainingsset wordt verder gesplitst in een federated trainingsset (80%) en een validatieset (20%).

3. **Vectorisatie:**
    - De tekstdata wordt getransformeerd naar TF-IDF representaties met behulp van `TfidfVectorizer`.

4. **Verdelen van de trainingsdata over clients:**
    - De federated trainingsdata wordt verdeeld over meerdere clients.

## Modeltraining

1. **Lokale modeltraining:**
    - Elk client traint een lokaal model (`SGDClassifier`) met zijn eigen data.
    - De gewichten (`coef_`) en intercepts (`intercept_`) van het globale model worden gebruikt als startpunt voor de lokale training (`warm_start=True`).

2. **Federated Averaging:**
    - Na elke trainingsronde worden de gewichten en intercepts van alle lokale modellen gemiddeld om het globale model bij te werken.
    - Het globale model wordt vervolgens geëvalueerd op de validatieset.

3. **Evaluatie:**
    - Na alle trainingsrondes wordt het globale model geëvalueerd op de testset.
    - Een centraal model wordt ook getraind op de volledige trainingsset en geëvalueerd op de testset voor vergelijking.


## Obstakels en Oplossingen
1. **Warm Start en Gewichten:**
    - Een obstakel was het correct instellen van de gewichten en intercepts van het globale model in de lokale modellen. Dit werd opgelost door warm_start=True te gebruiken.

2. **Classes_ Instellen:**
    - Een ander obstakel was dat het program niet werkt zonder global_model.classes_ = np.unique(y_train_fed) toe te voegen na het initialiseren van het globale model.

3. **Fluctuerende Nauwkeurigheid:**
    - De fluctuaties in nauwkeurigheid zijn een probleem waar geen duidelijke oplossing voor is.
