import cv2
import imutils
import numpy as np
import argparse


#TODO: stitch all images
#TODO: try using a bundle adjustment method and see the differences


def stitch_all_images(images, ratio=0.75, reprojThresh=4.0, showMatches=False):
    # images a list where each 2 is a pair of images?
    print "stitching all..."
    num_of_pairs = len(images) / 2
    results = []
    for i in xrange(num_of_pairs):
        imageB = images[i*2+1]
        imageA = images[i*2]
        (kpsA, featuresA) = detect_and_describe(imageA)
        (kpsB, featuresB) = detect_and_describe(imageB)
        M = match_key_points(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)
        if M is None:
            return None
        (matches, H, status) = M
        result = cv2.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
        result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
        if showMatches:
            vis = draw_matches(imageA, imageB, kpsA, kpsB, matches, status)
        results.append(result)
    return results


def stitch_images(images, ratio=0.75, reprojThresh=4.0, showMatches=False):
    print "stitching..."
    (imageB, imageA) = images
    (kpsA, featuresA) = detect_and_describe(imageA)
    (kpsB, featuresB) = detect_and_describe(imageB)
    M = match_key_points(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)
    if M is None:
        return None
    (matches, H, status) = M
    result = cv2.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
    print str(imageA.shape)
    print str(imageB.shape)
    print "shape: " + str(result.shape)
    result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
    if showMatches:
        vis = draw_matches(imageA, imageB, kpsA, kpsB, matches, status)
        return (result, vis)
    return result


def detect_and_describe(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    descriptor = cv2.xfeatures2d.SIFT_create()
    (kps, features) = descriptor.detectAndCompute(image, None)

    kps = np.float32([kp.pt for kp in kps])
    return (kps, features)


def match_key_points(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
    matcher = cv2.DescriptorMatcher_create("BruteForce")
    rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
    matches = []
    for m in rawMatches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            matches.append((m[0].trainIdx, m[0].queryIdx))
    if len(matches) > 4:
        ptsA = np.float32([kpsA[i] for (_, i) in matches])
        ptsB = np.float32([kpsB[i] for (i, _) in matches])
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)
        return (matches, H, status)
    return None


def draw_matches(imageA, imageB, kpsA, kpsB, matches, status):
    (hA, wA) = imageA.shape[:2]
    (hB, wB) = imageB.shape[:2]
    vis = np.zeros((max(hA, hB), wA+wB, 3), dtype="uint8")
    vis[0:hA, 0:wA] = imageA
    vis[0:hB, wA:] = imageB
    for ((trainIdx, queryIdx), s) in zip(matches, status):
        if s == 1:
            ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
            ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
            cv2.line(vis, ptA, ptB, (0, 255, 0), 1)
    return vis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", help="""mode""", type=int)
    args = parser.parse_args()
    test_single_pair = False
    if args.m == 0:
        test_single_pair = True
    if test_single_pair:
        imageA = cv2.imread("images/first.jpg")
        imageB = cv2.imread("images/second.jpg")
        imageA = imutils.resize(imageA, width=400)
        imageB = imutils.resize(imageB, width=400)
        (result, vis) = stitch_images([imageA, imageB], showMatches=True)
        cv2.imshow("Image A", imageA)
        cv2.imshow("Image B", imageB)
        cv2.imshow("Keypoint Matches", vis)
        cv2.imshow("Result", result)
        cv2.waitKey(0)
    else:
        images = []
        for i in xrange(18):
            file_name = "images/mcam_" + str(i+1) + "_scale_2.jpg"
            curr_image = cv2.imread(file_name)
            curr_image = imutils.resize(curr_image, width=400)
            images.append(curr_image)
        results = stitch_all_images(images, showMatches=False)
        for i in xrange(len(results)):
            print i
            cv2.imshow("Result" + str(i), results[i])
        cv2.waitKey(0)


if __name__  == "__main__":
    main()
