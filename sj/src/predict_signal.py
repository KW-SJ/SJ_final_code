from .video2biosig import predict_bio_sigs

def test():
    video_path="vid.mp4"
    res = predict_bio_sigs(video_path, 60, [5, 25])
    print(res)

if __name__ == '__main__':
    test()
    
    