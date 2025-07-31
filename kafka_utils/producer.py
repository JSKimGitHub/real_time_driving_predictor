from kafka import KafkaProducer
import json

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

def send_prediction_to_kafka(prediction, topic='fm-predictions'):
    producer.send(topic, {'action': int(prediction)})
    producer.flush()

