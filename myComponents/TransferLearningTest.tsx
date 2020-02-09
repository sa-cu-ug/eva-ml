
import React, { Fragment } from 'react';
import { SafeAreaView, StyleSheet, ScrollView, View, StatusBar, Image, Text, ImageSourcePropType } from 'react-native';

import * as tf from '@tensorflow/tfjs';
// import { fetch } from '@tensorflow/tfjs-react-native';
import * as jpeg from 'jpeg-js';
import * as imageGet from 'get-image-data';

interface ScreenProps {

}

interface ScreenState {
  prediction: any;
  predictionTime: number;
  isModelReady: boolean;
}

export class TransferLearningTest extends React.Component<
  ScreenProps,
  ScreenState
  > {
  constructor(props: ScreenProps) {
    super(props);
    this.state = {
	  prediction: [],
	  predictionTime: 0,
      isModelReady: false
    };
  }

  async componentDidMount() {

    await tf.ready();

	// Load model

	const modelJSON = require('./../converted-keras/standard_model_3/model.json');

	const loader = {
		load: async () => {
		  return {
			modelTopology: modelJSON.modelTopology,
			weightSpecs: modelJSON.specs,
			weightData: modelJSON.data,
		  };
		}
	}
	  
	const model = await tf.loadLayersModel(loader)

	// const model = await tf.loadLayersModel(modelJSON);
	//const model = await tf.loadGraphModel(require('./../converted-keras/model.json'));
	// const model = await tf.loadLayersModel(tf.io.httpRequest(require('./../converted-keras/model.json'), {fetch: fetch}));
	// const model = await tf.loadLayersModel('file://./../converted-keras/model.json');
	this.setState({
        isModelReady: true
	})

    // Read the image into a tensor
    const imageAssetPath = Image.resolveAssetSource(require('./../myAssets/dog1.jpg'));
    // const response = await fetch(imageAssetPath.uri, {}, { isBinary: true });
    // const rawImageData = await response.arrayBuffer();
	  // const imageTensor = this.imageToTensor(rawImageData);
	  const imageTensor = await this.loadLocalImage(imageAssetPath);

    // Classify the image.
    const start = Date.now();
    const prediction = model.predict(imageTensor);

    const end = Date.now();

    this.setState({
	  prediction,
	  predictionTime: end - start,
	});
	
	// Free memory
	tf.dispose([imageTensor]);
	
  }

  async loadLocalImage(filename): Promise<tf.Tensor3D> {
    return new Promise((res,rej)=>{
		imageGet(filename, (err, info) => {
			if(err){
				rej(err);
				return;
			}
			const image = tf.browser.fromPixels(info.data)
				.resizeNearestNeighbor([100, 100])
				// .toFloat
				// .expandDims();
				
			//   console.log(image, '127');
			res(image);
			});
		});
	}

  // https://gist.github.com/kevinvangelder/aa4dbc797bfb63e479f19597975a8a1c
  imageToTensor(rawImageData: ArrayBuffer): tf.Tensor3D {
    const TO_UINT8ARRAY = true;
    const { width, height, data } = jpeg.decode(rawImageData, TO_UINT8ARRAY);
    // Drop the alpha channel info for mobilenet
    const buffer = new Uint8Array(width * height * 3);
    let offset = 0; // offset into original data
    for (let i = 0; i < buffer.length; i += 3) {
      buffer[i] = data[offset];
      buffer[i + 1] = data[offset + 1];
      buffer[i + 2] = data[offset + 2];

      offset += 4;
    }

    return tf.tensor3d(buffer, [height, width, 3]);
  }

  renderPrediction() {
    const { prediction, predictionTime } = this.state;
    return (
      <View>
        <Text style={styles.resultTextHeader}>Results</Text>
        {prediction.map((pred, i) => {
          return (
            <View style={styles.prediction} key={i}>
              <Text style={styles.resultClass}>{pred.className}</Text>
              <Text style={styles.resultProb}>{pred.probability}</Text>
            </View>
          );
        })}
        <View style={styles.sectionContainer}>
          <Text>predictionTime: {predictionTime}</Text>
        </View>
      </View>
    );
  }

  render() {
    const { prediction, isModelReady } = this.state;

    return (
      <Fragment>
        <StatusBar barStyle='dark-content' />
        <SafeAreaView>
          <ScrollView
            contentInsetAdjustmentBehavior='automatic'
            style={styles.scrollView}
          >
            <View style={styles.body}>

              <View style={styles.sectionContainer}>
                {/* Title Area */}
                <View>
                  <Text style={styles.sectionTitle}>
                    Image Classification in React Native
                  </Text>
                </View>
                <View style={styles.resultArea}>
                    <Text>{isModelReady ? 'Ready' : 'Loading'}</Text>
                </View>
                {/* Result Area */}
                <View style={styles.resultArea}>
                  {prediction ? this.renderPrediction() : undefined}
                </View>
              </View>
            </View>
          </ScrollView>
        </SafeAreaView>
      </Fragment>
    );
  }
}

const styles = StyleSheet.create({
  scrollView: {
    backgroundColor: 'white'
  },
  body: {
    backgroundColor: 'white',
    marginBottom: 60
  },
  sectionContainer: {
    marginTop: 32,
    paddingHorizontal: 24
  },
  sectionTitle: {
    fontSize: 24,
    fontWeight: '600',
    color: 'black',
    marginBottom: 6
  },
  imageArea: {
    marginTop: 12,
    display: 'flex',
    flexDirection: 'column',
    justifyContent: 'center',
    alignItems: 'center'
  },
  resultArea: {
    marginLeft: 5,
    paddingLeft: 5,
    display: 'flex',
    flexDirection: 'column',
    justifyContent: 'center'
  },
  resultTextHeader: {
    fontSize: 21,
    fontWeight: 'bold',
    marginBottom: 12,
    marginTop: 12,
    textAlign: 'center'
  },
  prediction: {
    display: 'flex',
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center'
  },
  resultClass: {
    fontSize: 16,
    fontWeight: 'bold'
  },
  resultProb: {
    fontSize: 16,
    marginLeft: 5
  }
});