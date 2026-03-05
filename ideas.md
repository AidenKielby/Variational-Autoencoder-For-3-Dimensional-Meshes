from what i understand, there is an encoder and a decoder. The encoder encodes the input into vectors: a mean (μ) and a standard deviation (σ) for each dimension. 
the decoder tries to make the output as similar to the input as possible based on these vectors
once trained, you dont use the encoder anymore, and instead use the decoder with random numbers/vectors to generate the output

there are 3 ways i can think of doing this:
 - we use the encoder for shapes/meshes AND text so we can use text to generate meshes
 - we use the encoder for just shapes/meshes for now and worry about different inputs later
 - we do the seccond option, but train a small encoder to turn text into the same type of vectors
