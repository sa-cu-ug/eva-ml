global.Buffer = global.Buffer || require('buffer').Buffer
// https://stackoverflow.com/questions/48432524/cant-find-variable-buffer

import React, { PureComponent, Fragment } from 'react';
import { StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { SafeAreaView, ScrollView, StatusBar, Image, ImageSourcePropType } from 'react-native';
import { RNCamera } from 'react-native-camera';
import {PermissionsAndroid} from 'react-native';
import * as tf from '@tensorflow/tfjs';
import * as jpeg from 'jpeg-js';
import symbolicateStackTrace from 'react-native/Libraries/Core/Devtools/symbolicateStackTrace';

async function requestCameraPermission() {
  try {
	const granted = await PermissionsAndroid.request(
	  PermissionsAndroid.PERMISSIONS.CAMERA,
	  {
		title: 'Cool Photo App Camera Permission',
		message:
		  'Cool Photo App needs access to your camera ' +
		  'so you can take awesome pictures.',
		buttonNeutral: 'Ask Me Later',
		buttonNegative: 'Cancel',
		buttonPositive: 'OK',
	  },
	);
	if (granted === PermissionsAndroid.RESULTS.GRANTED) {
	  console.log('You can use the camera');
	} else {
	  console.log('Camera permission denied');
	}
  } catch (err) {
	console.warn(err);
  }
}

const PendingView = () => (
  <View
	style={{
	  flex: 1,
	  backgroundColor: 'lightgreen',
	  justifyContent: 'center',
	  alignItems: 'center',
	}}
  >
	<Text>Waiting for Model and Camera</Text>
  </View>
);

interface ScreenProps {

}

interface ScreenState {
	score: tf.Tensor<tf.Rank>;
	predictionTime: number;
	isModelReady: boolean;
	model: any;
}

export class BiometricRecognition extends React.Component<
ScreenProps,
ScreenState
> {

	constructor(props: ScreenProps) {
		super(props);
		this.state = {
		  	score: undefined,
		  	predictionTime: 0,
		  	isModelReady: false,
		  	model: undefined,
		};
	  }

	async componentDidMount() {

		// Hole Erlaubnis für Kamera-Nutzung ein
		await requestCameraPermission();

		// Lade tensorflow
		await tf.ready();

		// Pfad zum konvertierten Model
		const modelJSON = require('./../converted-keras/standard_model_3/model.json');

		// Custom-Loader für das konvertierte Model
		const loader = {
			load: async () => {
				return {
					modelTopology: modelJSON.modelTopology,
					weightSpecs: modelJSON.specs,
					weightData: modelJSON.data,
				};
			}
		}
	  
		// Lade konvertiertes Model
		const model = await tf.loadLayersModel(loader)

		this.setState({
			model: model,
			isModelReady: true
		})
	}

  render() {
	  const { score, predictionTime } = this.state;
	  // { predicted ? this.renderPrediction() : this.renderPhotoArea() }
	return (
		<Fragment>
			<View style={styles.container}>
				<RNCamera
				style={styles.preview}
				type={RNCamera.Constants.Type.front}
				flashMode={RNCamera.Constants.FlashMode.off}
				androidCameraPermissionOptions={{
					title: 'Permission to use camera',
					message: 'We need your permission to use your camera',
					buttonPositive: 'Ok',
					buttonNegative: 'Cancel',
				}}
				>
				{({ camera, status }) => {
					if (status !== 'READY') return <PendingView />;
					return (
					<View style={{ flex: 0, flexDirection: 'row', justifyContent: 'center' }}>

							{
								predictionTime > 0
								?
									// this.renderPrediction()
									<Text style={{ fontSize: 14 }}> { score.toString() } </Text>
								:
									<TouchableOpacity onPress={() => this.takePicture(camera)} style={styles.capture}>
										<Text style={{ fontSize: 14 }}> SNAP </Text>
									</TouchableOpacity>
							}

					</View>
					);
				}}
				</RNCamera>
			</View>
	  	</Fragment>
	);
  }

  // https://gist.github.com/kevinvangelder/aa4dbc797bfb63e479f19597975a8a1c
imageToTensor(rawImageData: ArrayBuffer): tf.Tensor3D {
    const TO_UINT8ARRAY = true;
    const { width, height, data } = jpeg.decode(rawImageData, TO_UINT8ARRAY);

    const buffer = new Uint8Array(width * height * 3);
    let offset = 0;
    for (let i = 0; i < buffer.length; i += 3) {
      buffer[i] = data[offset];
      buffer[i + 1] = data[offset + 1];
      buffer[i + 2] = data[offset + 2];

      offset += 4;
    }

    return tf.tensor3d(buffer, [height, width, 3]);
  }

  takePicture = async function(camera) {

	const { model } = this.state;

	// const options = { quality: 0.5, base64: true };
	const options = { base64: true };
	const data = await camera.takePictureAsync(options);
	// console.log(data.uri);

	// const imageTensor = await this.loadLocalImage(data.uri);

	// const imageAssetPath = Image.resolveAssetSource(require('./../myAssets/dog1.jpg'));
	// const response = await fetch(data.uri, {}, { isBinary: true });
	// const rawImageData = await response.arrayBuffer();
	// const imageTensor = this.imageToTensor(rawImageData);

	// http://jamesthom.as/blog/2018/08/13/serverless-machine-learning-with-tensorflow-dot-js/

	const decodeImage = source => {
		const buf = Buffer.from(source, 'base64')
		const pixels = jpeg.decode(buf, true);
		return pixels;
	}

	const imageTensor = decodeImage(data.base64);

	const imageTensorResized = tf.browser.fromPixels(imageTensor)
		.resizeNearestNeighbor([100, 100])

	// Batch Size = 1, weil nur 1 Bild übergeben wird, Size, 3 Farbkanäle
	const outShape: [number,number, number, number] = [1, 100, 100, 3];
	const finalTensor = tf.tensor4d(imageTensorResized.dataSync(), outShape, 'float32');

	const start = Date.now();
	const prediction = model.predict(finalTensor);
	const score = (prediction as tf.Tensor<tf.Rank>).dataSync()[0];
	const end = Date.now();

	this.setState({
		score: score,
		predictionTime: end - start,
	});

	tf.dispose([finalTensor]);
  };
}

const styles = StyleSheet.create({
  container: {
	flex: 1,
	flexDirection: 'column',
	backgroundColor: 'black',
  },
  preview: {
	flex: 1,
	justifyContent: 'flex-end',
	alignItems: 'center',
  },
  capture: {
	flex: 0,
	backgroundColor: '#fff',
	borderRadius: 5,
	padding: 15,
	paddingHorizontal: 20,
	alignSelf: 'center',
	margin: 20,
  },
});
