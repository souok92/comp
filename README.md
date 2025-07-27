# comp

## connect
- 베라소닉스 -> 파이썬 -> 무조코 워크플로우
  1. saveFrameL12_5withButtons_modified.m
     saveImg2.m이랑 같이 사용, 버튼 2개 추가.
     연속저장버튼은 버퍼에 지속적으로 프레임 저장하는 버튼 / 6개짜리 버튼은 버튼 누르면 분류까지 해서 저장
     나머지는 자동화해뒀으니 넘어가고
     바탕화면에 Gest, Buffer 폴더 2개가 생성될거고 Buffer는 파이썬 연동용, Gest는 분류 데이터 획득용

  2. LiveMujoco.py
     기존의 폴더에서 랜덤으로 읽어오는 사진을 Buffer 폴더에서 읽어오는 방식으로 진행
