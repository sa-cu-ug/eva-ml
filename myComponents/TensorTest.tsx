import React, { Component } from 'react';
import { Text, View } from 'react-native';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-react-native';

interface IMyProps {

}

interface IMyState {
    isTfReady: boolean;
}

export class TensorTest extends Component<IMyProps, IMyState> {

    constructor(props: IMyProps) {
        super(props);
        this.state = {
            isTfReady: false,
        };
    }

    async componentDidMount() {

        console.log('componentDidMount')

        // Wait for tf to be ready.
        await tf.ready();
        // Signal to the app that tensorflow.js can now be used.
        console.log('isTfReady: true');
        this.setState({
            isTfReady: true,
        });
    }

    render() {

        const { isTfReady } = this.state;

        let text;

        if(isTfReady) {
            text = <Text>Ready</Text>
        } else {
            text = <Text>Loading..</Text>
        }

        return (
            <View>
                {text}
            </View>
        )
    }
}

