import cv2

def stitch_images(images, ratio =0.75, reprojThresh=4.0, showMatches=False):
    print "stitching..."
    (imageB, imageA) = images
    (kpsA, featuresA) = detect_and_describe(imageA)
    (kpsB, featuresB) = detect_and_describe(imageB)
    M = matchKeyPoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)
    if M is None:
        return None
    (matches, H, status) = M
    result = cv2.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
    result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
    if showMatches:
        vis = drawMatches(imageA, imageB, kpsA, kpsB, matches, status)
        return (result, vis)
    return result


def detect_and_describe(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    descriptor = cv2.xfeatures2d.SIFT_create()
    (kps, features) = descriptor.detectAndCompute(image, None)

    kps = np.float32([kp.pt for kp in kps])
    return (kps, features)    



def main():
    stitch_images()


if __name__  == "__main__":
    main()
