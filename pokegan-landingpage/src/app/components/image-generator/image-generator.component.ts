import { Component, OnInit } from '@angular/core';
import * as tf from '@tensorflow/tfjs';

@Component({
  selector: 'app-image-generator',
  templateUrl: './image-generator.component.html',
  styleUrls: ['./image-generator.component.scss'],
})
export class ImageGeneratorComponent implements OnInit {
  model: tf.LayersModel = undefined;
  generatedPokemons: string[] = [];

  async ngOnInit(): Promise<void> {
    this.model = await tf.loadLayersModel('assets/generator-model/model.json');
    this.onClickCreatePokemon()
  }

  onClickCreatePokemon(): void {
    this.generatedPokemons = [];
    for (let i = 0; i < 16; i++) {
      const generatedPokemonTensor: any = this.model.predict(
        tf.randomNormal([1, 100], 0, 1)
      );
      const scaledGeneratedPokemonArray = tf
        .add(
          tf.mul(generatedPokemonTensor, tf.scalar(128)).toInt(),
          tf.scalar(128)
        )
        .flatten()
        .arraySync();

      const width = 64;
      const height = 64;
      const channels = 4;

      const buffer = new Uint8Array(width * height * channels);
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const index = (y * width + x) * 4;
          buffer[index] = scaledGeneratedPokemonArray[(y * width + x) * 3];
          buffer[index + 1] =
            scaledGeneratedPokemonArray[(y * width + x) * 3 + 1];
          buffer[index + 2] =
            scaledGeneratedPokemonArray[(y * width + x) * 3 + 2];
          buffer[index + 3] = 255;
        }
      }

      const canvas = document.createElement('canvas');
      const context = canvas.getContext('2d');

      canvas.width = width;
      canvas.height = height;

      const imageData = context.createImageData(width, height);
      imageData.data.set(buffer);

      context.putImageData(imageData, 0, 0);

      const dataURL = canvas.toDataURL();

      this.generatedPokemons.push(dataURL);
    }
  }
}
