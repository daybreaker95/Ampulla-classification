# 폴더 내의 모든 파일의 이름을 '등록번호', ' ', '이름'만 남기고 없애줍니다.

# 파일 이름을 변경할 폴더 경로
folder_path = '/content/drive/MyDrive/박도현 교수님 연구실/췌액 채취 속도/collection speed classification'  # 여기에 원하는 폴더 경로를 입력하세요.

# 폴더 내의 모든 파일 이름 변경
for file_name in os.listdir(folder_path):
    # 파일의 전체 경로
    old_file_path = os.path.join(folder_path, file_name)

    # 파일이 파일인지 확인
    if os.path.isfile(old_file_path):
        # 파일 이름에서 등록번호와 이름 추출
        parts = file_name.split(' ')
        if len(parts) >= 2:
            # 등록번호와 이름만 추출
            new_file_name = f"{parts[0]} {parts[1]}"
            new_file_path = os.path.join(folder_path, new_file_name)

            # 파일 이름 변경
            os.rename(old_file_path, new_file_path)
            print(f"변경됨: {old_file_path} -> {new_file_path}")
        else:
            print(f"형식이 맞지 않음: {file_name}")

##############################

# 파일 이름을 "'등록번호' '이름' 'collection_rate_class'"으로 변경합니다.

import os
import pandas as pd

# 파일 이름을 변경할 폴더 경로
folder_path = '/content/drive/MyDrive/박도현 교수님 연구실/췌액 채취 속도/collection speed classification'  # 여기에 원하는 폴더 경로를 입력하세요.

# 폴더 내의 모든 파일 이름 변경
for file_name in os.listdir(folder_path):
    old_path = os.path.join(folder_path, file_name)
    # 파일 이름에서 등록번호 추출
    file_reg_num = file_name.split(' ')[0]  # 파일 이름의 등록번호
    file_reg_num = str(file_reg_num)  # file_reg_num을 문자열로 변환
    df['등록번호'] = df['등록번호'].astype(str)  # df['등록번호']를 문자열로 변환


    # 등록번호와 일치하는 인덱스 찾기
    matching_index = df[df['등록번호'] == file_reg_num].index

    if not matching_index.empty:
        # collection_rate_class 값 추출
        collection_rate_class = df.at[matching_index[0], 'collection_rate_class']  # 인덱스를 사용하여 값 추출
        # 새로운 파일 이름 생성
        new_file_name = f"{file_name} {collection_rate_class}"  # 기존 파일 이름에 추가
        new_path = os.path.join(folder_path, new_file_name)

        # 파일 이름 변경
        os.rename(old_path, new_path)
        print(f"변경됨: {old_path} -> {new_path}")
    else:
        print(f"일치하는 등록번호 없음: {file_reg_num}")

##############################

# 파일을 train_test_split method를 통해 'train', 'val' 폴더에 분류합니다. stratified by 'collection_rate_class'.

from sklearn.model_selection import train_test_split
import os
import pandas as pd
import shutil

data_dir= '/content/drive/MyDrive/박도현 교수님 연구실/췌액 채취 속도/collection speed classification'

# 데이터 디렉토리에서 파일 이름 가져오기
file_names = os.listdir(data_dir)

# 파일 이름에서 정보를 추출하고 DataFrame 생성
file_info = []
for file_name in file_names:
    if os.path.isfile(os.path.join(data_dir, file_name)):
        # '등록번호' '이름' 'collection_rate_class'로 구분
        parts = file_name.split(' ')
        if len(parts) == 3:  # 형식이 올바른 파일만 사용
            reg_num, name, collection_rate_class = parts[0], parts[1], parts[2].split('.')[0]  # 확장자 제거
            file_info.append({'file_name': file_name, 'reg_num': reg_num, 'name': name, 'collection_rate_class': collection_rate_class})

# DataFrame 생성
df = pd.DataFrame(file_info)

# DataFrame 확인
print(df.head())

# 'collection_rate_class' 컬럼이 있는지 확인하고, train, val 분할
if 'collection_rate_class' in df.columns:
    train_df, val_df = train_test_split(df, test_size=0.3, stratify=df['collection_rate_class'], random_state=42)
else:
    print("DataFrame에 'collection_rate_class' 컬럼이 없습니다.")

# train, val 디렉토리 경로 설정
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')

def copy_files(file_df, target_dir, extension='.jpg'):
    os.makedirs(target_dir, exist_ok=True)
    for _, row in file_df.iterrows():
        src_path = os.path.join(data_dir, row['file_name'])
        dst_path = os.path.join(target_dir, row['file_name'])
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy(src_path, dst_path)

# train, val 디렉토리에 파일 복사
copy_files(train_df, train_dir)
copy_files(val_df, val_dir)

print("파일 분류가 완료되었습니다.")

###########################################

import os
import shutil
import pandas as pd

# 데이터 디렉토리 설정
data_dir = '/content/drive/MyDrive/박도현 교수님 연구실/췌액 채취 속도/collection speed classification'

# train 및 val 폴더 경로 설정
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')

# 파일 이동을 위한 함수 정의
def organize_files_by_class(directory):
    # 파일 이름 형식 ['등록번호', '이름', 'collection_rate_class']
    file_names = os.listdir(directory)

    # 파일 이름에서 정보 추출하고 DataFrame 생성
    file_info = []
    for file_name in file_names:
        if os.path.isfile(os.path.join(directory, file_name)):
            # '등록번호', '이름', 'collection_rate_class'로 구분
            parts = file_name.split(' ')
            reg_num, name = parts[0], parts[1]
            collection_rate_class = parts[2].split('.')[0]  # 확장자를 제외한 collection_rate_class만 추출
            file_info.append({
                'file_name': file_name,  # 파일 전체 이름을 저장하여 확장자 포함
                'reg_num': reg_num,
                'name': name,
                'collection_rate_class': collection_rate_class
            })

    # DataFrame 생성
    df = pd.DataFrame(file_info)

    # 각 collection_rate_class별로 하위 폴더 생성 후 파일 이동
    for _, row in df.iterrows():
        target_class_dir = os.path.join(directory, row['collection_rate_class'])
        os.makedirs(target_class_dir, exist_ok=True)

        # 파일 이동
        src_path = os.path.join(directory, row['file_name'])
        dst_path = os.path.join(target_class_dir, row['file_name'])
        shutil.copy(src_path, dst_path)

    print(f"{directory} 폴더 내의 이미지 파일들이 collection_rate_class별로 폴더에 저장되었습니다.")

# train 및 val 폴더 내 파일 정리
organize_files_by_class(train_dir)
organize_files_by_class(val_dir)
