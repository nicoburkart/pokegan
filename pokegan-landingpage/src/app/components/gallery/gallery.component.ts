import { Component, OnInit } from '@angular/core';

@Component({
  selector: 'app-gallery',
  templateUrl: './gallery.component.html',
  styleUrls: ['./gallery.component.scss']
})
export class GalleryComponent implements OnInit {

  gallery_images = 32
  gallery_indizes = []

  constructor() { }

  ngOnInit(): void {
    for(let i=1; i<=this.gallery_images; i++)
      this.gallery_indizes[i-1] = i
    }

}
