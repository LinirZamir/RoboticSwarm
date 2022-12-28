import paho.mqtt.client as mqtt
from paho.mqtt.subscribeoptions import SubscribeOptions
from types import SimpleNamespace
import main 

# Replace with your AWS IoT endpoint
IoT_protocol_name = "x-amzn-mqtt-ca"
iot_endpoint = "a3q0a3wnhomfgs-ats.iot.us-east-1.amazonaws.com"
private_key = "keys/private.pem.key"
certificate = "keys/certificate.pem.crt"
root_ca = "keys/AmazonRootCA1.pem"

# when the MQTT client connects to the broker
def on_connect(client, userdata, flags, rc, properties=None):
    client.subscribe('bees/heartbeat', options=SubscribeOptions(noLocal=True))
    ## result = userdata.session.execute(f"INSERT INTO bee_robots.bee_table (id) VALUES ({userdata.id})")

# Define the message callback function
def on_message(client, userdata, msg, properties=None):
    topic = msg.topic
    payload = msg.payload.decode()
    if topic == 'bees/heartbeat':
        #logger.debug("Received message:{} on topic {}".format(msg.payload,msg.topic))
        found = False
        parts = payload.split(' ')
        robot_id = parts[0]
        robot_state = parts[1]
        robot_x = parts[2]
        robot_y = parts[3]
        for robot in main.robot_list:
            if robot.id == robot_id:
                robot.pos_x=float(robot_x)
                robot.pos_y=float(robot_y)
                robot.state=0
                found = True
        if not found:
            main.robot_list.append(SimpleNamespace(id=robot_id, state=0, pos_x=float(robot_x), pos_y=float(robot_y)))
    else:
        #logger.info("Received message:{} on topic {}".format(msg.payload,msg.topic))
        return

def on_disconnect(client, userdata, rc, properties=None):
    client.loop_stop(force=False)
    if rc != 0:
        print("Unexpected disconnection.")
    else:
        print("Disconnected")

# Create an MQTT client
def mqtt_run(client):
    try:
        # Define the MQTT topics for publishing and subscribing
        client.on_message = on_message
        client.on_connect = on_connect
        client.on_disconnect = on_disconnect

        client.connect("54.173.136.48", port=1883, keepalive=60, bind_address="", bind_port=0,
clean_start=mqtt.MQTT_CLEAN_START_FIRST_ONLY, properties=None)

        # Start the MQTT client loop in a separate thread
        client.loop_forever()
                    
    except Exception as e:
        print("ERROR!")