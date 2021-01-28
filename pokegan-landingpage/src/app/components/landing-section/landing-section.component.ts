import { Component, HostListener, OnInit } from '@angular/core';

@Component({
  selector: 'app-landing-section',
  templateUrl: './landing-section.component.html',
  styleUrls: ['./landing-section.component.scss']
})
export class LandingSectionComponent implements OnInit {

  constructor() { }

  ngOnInit(): void {
  }

  showHideScrollIndicator(): void {
    const arrow = document.getElementsByClassName(
      'arrow-down'
    )[0] as HTMLElement;
    arrow.style.opacity = `${1 - window.scrollY / 300}`;
  }

  @HostListener('window:scroll', ['$event'])
  onScrollEvent($event: Event): void {
    this.showHideScrollIndicator();
  }

}
