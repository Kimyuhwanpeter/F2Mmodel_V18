# F2Mmodel_V18
* Weight standardization을 각 layer에 모두 적용
* 전체적인 모델 구조는 F2M_model_V16과 동일 (중간에 high pass filter는 빼고 shapren filter를 대체하였음)
* WS는 weight로부터 얻은 loss의 gradient를 줄이는 효과를 가지고 있고 SAM은 loss의 경사하강의 sharpeness를 극대화 하는 효과를 가지고 있기 때문에, 두 기법을 섞으면 좋은 기대 효과를 볼 수 있을것으로 판단
