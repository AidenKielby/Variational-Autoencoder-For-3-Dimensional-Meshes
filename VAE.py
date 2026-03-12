from Network import NeuralNetwork
import numpy as np

class Encoder:
    def __init__(self, inputNeurons: int, hiddenLayers: int, neuronsPerHiddenLayer: int, outputNeurons: int):
        self.network = NeuralNetwork(inputNeurons, hiddenLayers, neuronsPerHiddenLayer, outputNeurons)
        self.mean = np.zeros(outputNeurons//2)
        self.logvar = np.zeros(outputNeurons//2)
        self.std = np.zeros(outputNeurons//2)

    def forward(self, input):
        self.network.addInputs(input)
        self.network.forwardPass()
        output = self.network.neurons[-1]

        half = len(output) // 2
        self.mean = output[:half]
        self.logvar = output[half:]

        self.std = np.exp(0.5 * self.logvar)

        return self.mean, self.std
    
class Decoder:
    def __init__(self, inputNeurons: int, hiddenLayers: int, neuronsPerHiddenLayer: int, outputNeurons: int):
        self.network = NeuralNetwork(inputNeurons, hiddenLayers, neuronsPerHiddenLayer, outputNeurons)
    
    def forward(self, input):
        self.network.addInputs(input)
        self.network.forwardPass()
        output = self.network.neurons[-1]

        return output

class VAE:
    def __init__(self, inputSize: int, hiddenLayers: int, neuronsPerHiddenLayer: int, outputSize: int):
        self.encoder = Encoder(inputSize, hiddenLayers, neuronsPerHiddenLayer, outputSize)
        self.decoder = Decoder(outputSize//2, hiddenLayers, neuronsPerHiddenLayer, outputSize)

    def forward(self, input):
        mu, std = self.encoder.forward(input)
        z = mu+std * np.random.randn(*mu.shape)

        decoder_out = self.decoder.forward(z)

        return decoder_out, mu, self.encoder.logvar
        