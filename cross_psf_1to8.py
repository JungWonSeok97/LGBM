# psf_1to2.py
from cross_psf1 import run_cross_psf1
from cross_psf2 import run_cross_psf2
from cross_psf3 import run_cross_psf3
from cross_psf4 import run_cross_psf4
from cross_psf5 import run_cross_psf5
from cross_psf6 import run_cross_psf6
from cross_psf7 import run_cross_psf7
from cross_psf8 import run_cross_psf8

def main():
    print("=" * 60)
    print("▶ PSF1 (작업부하) 결과")
    print("=" * 60)
    run_cross_psf1()

    print("\n\n" + "=" * 60)
    print("▶ PSF2 (장비 및 기기) 결과")
    print("=" * 60)
    run_cross_psf2()

    print("\n\n" + "=" * 60)
    print("▶ PSF3 (직무 절차) 결과")
    print("=" * 60)
    run_cross_psf3()

    print("\n\n" + "=" * 60)
    print("▶ PSF4 (교육/훈련) 결과")
    print("=" * 60)
    run_cross_psf4()

    print("\n\n" + "=" * 60)
    print("▶ PSF5 (의사소통) 결과")
    print("=" * 60)
    run_cross_psf5()

    print("\n\n" + "=" * 60)
    print("▶ PSF6 (환경) 결과")
    print("=" * 60)
    run_cross_psf6()

    print("\n\n" + "=" * 60)
    print("▶ PSF7 (역할 분담/작업 계획/협업/감독) 결과")
    print("=" * 60)
    run_cross_psf7()

    print("\n\n" + "=" * 60)
    print("▶ PSF8 (인지적 복잡성 / 의사결정) 결과")
    print("=" * 60)
    run_cross_psf8()

if __name__ == "__main__":
    main()
