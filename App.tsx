/**
 * Sample React Native App
 * https://github.com/facebook/react-native
 *
 * Generated with the TypeScript template
 * https://github.com/emin93/react-native-template-typescript
 *
 * @format
 */

import React, {Fragment} from 'react';
import {
  StyleSheet,
  View,
  Text,
} from 'react-native';

// import { TensorTest } from './myComponents/TensorTest';
// import { ImageRecognitionTest } from './myComponents/ImageRecognitionTest';
// import { TransferLearningTest } from './myComponents/TransferLearningTest';
import { BiometricRecognition } from './myComponents/BiometricRecognition';

const App = () => {
  return (
    // <View style={styles.container}>
    //   {/* <Text>Test</Text> */}
    //   <TensorTest></TensorTest>
    // </View>
    // <TensorTest></TensorTest>
    // <ImageRecognitionTest></ImageRecognitionTest>
    // <TransferLearningTest></TransferLearningTest>
    <BiometricRecognition></BiometricRecognition>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
  },
});

export default App;
