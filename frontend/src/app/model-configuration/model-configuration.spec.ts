import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ModelConfiguration } from './model-configuration';

describe('ModelConfiguration', () => {
  let component: ModelConfiguration;
  let fixture: ComponentFixture<ModelConfiguration>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ModelConfiguration]
    })
    .compileComponents();

    fixture = TestBed.createComponent(ModelConfiguration);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
