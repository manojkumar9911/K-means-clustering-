import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Sample text documents
documents = [
"Regular physical activity can improve your overall health. It helps reduce the risk of chronic diseases like heart disease, diabetes, and obesity. Aim for at least 150 minutes of moderate-intensity exercise per week.",
    "Proper nutrition is essential for a healthy lifestyle. A balanced diet rich in fruits, vegetables, lean proteins, and whole grains can boost your energy, support weight management, and improve your immune system.",
    "Mental health is as important as physical health. It's essential to prioritize self-care, seek help when needed, and reduce stigma to create a healthier society.",
    "Quality sleep is crucial for your health. Aim for 7-9 hours of restful sleep each night to recharge your body and mind.",
    "High stress levels can harm your health. Practice relaxation techniques like deep breathing, meditation, and exercise to manage stress effectively.",
    "Hydration is key to staying healthy. Drink enough water daily to support your body's functions and maintain optimal health.",
    "Regular check-ups with your healthcare provider are essential for early disease detection and preventive care.",
    "The benefits of a plant-based diet include lower cholesterol levels, reduced risk of heart disease, and better weight management.",
    "Quitting smoking is one of the best things you can do for your health. It reduces the risk of cancer, heart disease, and respiratory problems.",
    "Sunscreen is crucial to protect your skin from harmful UV rays and reduce the risk of skin cancer.",
    "Vaccinations are a critical part of public health, helping to prevent the spread of infectious diseases and protect vulnerable populations.",
    "Regular dental check-ups are essential for maintaining good oral health and preventing dental issues.",
    "Physical therapy is often recommended for rehabilitation after injuries, helping individuals regain their strength and mobility.",
    "Yoga and meditation are valuable practices for reducing stress and promoting mental and physical well-being.",
    "The importance of a well-balanced breakfast in providing energy and focus throughout the day cannot be overstated.",
    "Adequate vitamin D intake is vital for maintaining strong bones and a healthy immune system.",
    "Chronic sleep deprivation can lead to a host of health problems, including cognitive impairment and mood disorders.",
    "The Mediterranean diet, rich in olive oil, fruits, vegetables, and fish, is associated with a reduced risk of heart disease.",
    "Regular health screenings, such as mammograms and colonoscopies, can detect cancer and other conditions early.",
    "Maintaining a healthy body weight through a combination of diet and exercise is crucial for long-term health.",
    "The thrill of watching live sports events in stadiums, surrounded by passionate fans, is an unforgettable experience.",
    "Table tennis, or ping pong, is a fast-paced sport that requires quick reflexes and precision.",
    "Ice hockey is a dynamic and physically demanding sport that captivates audiences with its speed and intensity.",
    "Gymnastics is a sport that combines strength, flexibility, and grace, with athletes performing breathtaking routines.",
    "The mental toughness of athletes often determines their success, making psychology an important aspect of sports.",
    "Running marathons is a challenging but rewarding achievement, testing endurance and determination.",
    "The Paralympic Games showcase the incredible athleticism and resilience of para-athletes on the world stage.",
    "Skateboarding, known for its creativity and daring tricks, is a popular action sport among young enthusiasts.",
    "Swimming is an excellent full-body workout that improves cardiovascular fitness and builds muscular strength.",
    "Boxing requires a unique combination of speed, power, and strategy, making it a popular combat sport.",
    "Skiing and snowboarding are thrilling winter sports enjoyed by those seeking adventure on the slopes.",
    "The mental aspect of sports coaching plays a significant role in an athlete's performance and development.",
    "Cycling is both a competitive sport and a recreational activity that offers health benefits and environmental advantages.",
    "The camaraderie and teamwork in team sports like soccer and basketball create strong bonds among players.",
    "Sports journalism and broadcasting provide fans with in-depth analysis and coverage of their favorite games.",
    "Volleyball is an exciting sport with fast-paced action and spectacular spikes, popular in both indoor and beach variations.",
    "Sports nutrition is crucial for optimizing an athlete's performance and recovery.",
    "Track and field events showcase incredible human feats of speed, strength, and endurance in competitions.",
    "Cheerleading combines athleticism, teamwork, and spirit in supporting other sports teams.",
    "Swimming is an essential life skill, providing water safety and health benefits for individuals of all ages."

]

# Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

# Perform K-Means clustering
k = 2  # Number of clusters
kmeans = KMeans(n_clusters=k, random_state=0)
kmeans.fit(tfidf_matrix)

# Get the cluster labels
cluster_labels = kmeans.labels_

# Print the cluster assignments for each document
# Print the cluster assignments for each document with commas
print(", ".join(map(str, cluster_labels)))


# You can also access the cluster centers if needed
cluster_centers = kmeans.cluster_centers_


