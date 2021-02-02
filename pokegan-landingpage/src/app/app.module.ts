import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';

import { AppComponent } from './app.component';
import { LandingSectionComponent } from './components/landing-section/landing-section.component';
import { HeadlineComponent } from './components/headline/headline.component';
import { GalleryComponent } from './components/gallery/gallery.component';
import { ButtonComponent } from './components/button/button.component';
import { ImageGeneratorComponent } from './components/image-generator/image-generator.component';
import { ContactComponent } from './components/contact/contact.component';
import { TrainingInfoComponent } from './components/training-info/training-info.component';

@NgModule({
  declarations: [
    AppComponent,
    LandingSectionComponent,
    HeadlineComponent,
    GalleryComponent,
    ButtonComponent,
    ImageGeneratorComponent,
    ContactComponent,
    TrainingInfoComponent
  ],
  imports: [
    BrowserModule
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
