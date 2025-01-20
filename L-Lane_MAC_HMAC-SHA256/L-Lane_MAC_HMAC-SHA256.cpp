#include <iostream>
#include <vector>
#include <iomanip>
#include <immintrin.h>
#include <chrono> // 時間の計測に使用
#include <thread>

void L_Lane_HMAC(std::vector<uint8_t>& data, std::vector<uint8_t>& key, std::vector<uint8_t>& hmac);
void hmac_sha256(std::vector<uint8_t>& data, std::vector<uint8_t>& paddedkey, std::vector<uint8_t>& hmac);
std::vector<uint8_t> sha256_shani(std::vector<uint8_t>& data);

std::vector<uint8_t> set_data0x00(size_t length) {
    std::vector<uint8_t> buffer(length, 0x01);
    return buffer;
}

int main() {
    std::vector<uint8_t> key = set_data0x00(64); // キー
    std::vector<uint8_t> data = set_data0x00(1); // 入力データ
    std::vector<uint8_t> hmac;
    int rep_cnt = 1; // 計測回数
    int func_cnt = 1; // 関数の繰り返し回数

    for (int i = 0; i < rep_cnt; i++) {
        // ↓ここから時間を計測する
        auto start = std::chrono::high_resolution_clock::now(); // 開始時間を記録

        for (int i = 0; i < func_cnt; i++) {
            L_Lane_HMAC(data, key, hmac);
        }

        auto end = std::chrono::high_resolution_clock::now(); // 終了時間を記録
        //↑計測ここまで

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start); // 経過時間を計算
        std::cout << duration.count() << std::endl;
    }

    return 0;
}

void L_Lane_HMAC(std::vector<uint8_t>& data, std::vector<uint8_t>& key, std::vector<uint8_t>& hmac){
    // キーのパディング処理
    size_t key_length = key.size();
    static std::vector<uint8_t> paddedkey(64);
    if (key_length > 64) {
        auto hashed_key = sha256_shani(key);
        std::copy(hashed_key.begin(), hashed_key.end(), paddedkey.begin());
        std::fill(paddedkey.begin() + 32, paddedkey.end(), 0x00);
    }
    else {
        std::copy(key.begin(), key.end(), paddedkey.begin());
        if (key_length != 64) {
            std::fill(paddedkey.begin() + key_length, paddedkey.end(), 0x00);
        }
    }

    //データをレーン1~3に分ける
    size_t block_num = data.size() / 128;
    size_t buf_size = data.size() % 128 == 0 ? block_num * 64 : (block_num + 1) * 64;
    if (data.size() == 0) buf_size = 64;
    std::vector<uint8_t> L1_data(buf_size), L2_data(buf_size), L3_data(buf_size);
    __m512i odd, even;
    int i;

    for (i = 0; i < block_num; ++i) {
        odd = _mm512_loadu_si512(data.data() + (i * 128));
        even = _mm512_loadu_si512(data.data() + (i * 128) + 64);

        _mm512_storeu_si512(L1_data.data() + (i * 64), odd);
        _mm512_storeu_si512(L2_data.data() + (i * 64), even);
        _mm512_storeu_si512(L3_data.data() + (i * 64), _mm512_xor_si512(odd, even));
    }

    if ((data.size() / 64 % 2) == 0) {
        std::copy(data.begin() + (block_num * 128), data.end(), L1_data.data() + (i * 64));
        std::copy(data.begin() + (block_num * 128), data.end(), L3_data.data() + (i * 64));
    }
    else {
        odd = _mm512_loadu_si512(data.data() + (i * 128));
        _mm512_storeu_si512(L1_data.data() + (i * 64), odd);
        if(data.size() % 64 != 0) std::copy(data.begin() + (block_num * 128) + 64, data.end(), L2_data.data() + (i * 64));
        _mm512_storeu_si512(L3_data.data() + (i * 64), _mm512_xor_si512(odd, _mm512_loadu_si512(L2_data.data() + (i * 64))));
    }

    //それぞれのレーンでMAC生成（並列処理）
    static std::vector<uint8_t> L1_MAC(32), L2_MAC(32), L3_MAC(32);
    std::thread thread1(hmac_sha256, std::ref(L1_data), std::ref(paddedkey), std::ref(L1_MAC));
    std::thread thread2(hmac_sha256, std::ref(L2_data), std::ref(paddedkey), std::ref(L2_MAC));
    std::thread thread3(hmac_sha256, std::ref(L3_data), std::ref(paddedkey), std::ref(L3_MAC));
    thread1.join();
    thread2.join();
    thread3.join();

    //3つのMACを結合
    static std::vector<uint8_t> last_data(96);
    std::copy(L1_MAC.begin(), L1_MAC.end(), last_data.begin());
    std::copy(L2_MAC.begin(), L2_MAC.end(), last_data.begin() + 32);
    std::copy(L3_MAC.begin(), L3_MAC.end(), last_data.begin() + 64);

    //その値でMAC生成
    hmac_sha256(last_data, paddedkey, hmac);
}

//HMAC-SHA-256を計算する関数
void hmac_sha256(std::vector<uint8_t>& data, std::vector<uint8_t>& paddedkey, std::vector<uint8_t>& hmac) {
    //H((K0 ⊕ opad) || H((K0 ⊕ ipad) || text))の計算
    thread_local std::vector<uint8_t> ipad(64 + data.size());
    thread_local std::vector<uint8_t> opad(96);
    thread_local std::vector<uint8_t> ihash(32);

    thread_local size_t predata_size = data.size();
    if (predata_size < data.size()) {
        ipad.resize(64 + data.size());
        predata_size = data.size();
    }

    std::fill(ipad.begin(), ipad.begin() + 64, 0x36);
    std::fill(opad.begin(), opad.begin() + 64, 0x5C);

    for (size_t i = 0; i < 64; ++i) {
        ipad[i] ^= paddedkey[i];
        opad[i] ^= paddedkey[i];
    }

    std::copy(data.begin(), data.end(), ipad.begin() + 64);
    ihash = sha256_shani(ipad);
    std::copy(ihash.begin(), ihash.end(), opad.begin() + 64);

    // ハッシュ関数を掛けてreturn
    hmac = sha256_shani(opad);
}

void oneblock_sha256(__m128i* front_hash, __m128i* back_hash, __m128i W[]) {
    // SHA-256のラウンド定数
    static const uint32_t CONST_K[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
    };
    __m128i ABEF, CDGH, X, Y, Ki;
    int i, j;

    ABEF = _mm_load_si128(front_hash);
    CDGH = _mm_load_si128(back_hash);

    // 0～15 Wordの処理
    for (i = 0; i < 4; ++i) {
        // CONST_Kとmsgをビッグエンディアンで取り出す(128bit単位)
        Ki = _mm_load_si128((__m128i const*)CONST_K + i);

        // W+Kの値を出す(前半64bitを使う)
        X = _mm_add_epi32(W[i], Ki); // 128bit分のW+K
        Y = _mm_shuffle_epi32(X, 0x0E); // X0の後半64bitを前に持ってくる

        // 計４回分(128bit)のハッシュ計算
        CDGH = _mm_sha256rnds2_epu32(CDGH, ABEF, X);
        ABEF = _mm_sha256rnds2_epu32(ABEF, CDGH, Y);
    }
    // 16～63 Wordの処理
    for (j = 1; j < 4; ++j) {
        for (i = 0; i < 4; ++i) {
            // CONST_Kを取り出す
            Ki = _mm_load_si128((__m128i const*)CONST_K + 4 * j + i);

            // メッセージスケジュール作成（前半）
            X = _mm_sha256msg1_epu32(W[i], W[(i + 1) % 4]);

            // X0にW(t-7)を加える
            Y = _mm_alignr_epi8(W[(i + 3) % 4], W[(i + 2) % 4], 4);
            X = _mm_add_epi32(X, Y);

            // メッセージスケジュール作成（後半）
            W[i] = _mm_sha256msg2_epu32(X, W[(i + 3) % 4]);

            // W+Kの値を出す(前半64bitを使う)
            X = _mm_add_epi32(W[i], Ki);
            Y = _mm_shuffle_epi32(X, 0x0E);

            // 計４回分(128bit)のハッシュ計算
            CDGH = _mm_sha256rnds2_epu32(CDGH, ABEF, X);
            ABEF = _mm_sha256rnds2_epu32(ABEF, CDGH, Y);
        }
    }
    // ハッシュ値の更新
    _mm_store_si128(front_hash, _mm_add_epi32(ABEF, *front_hash));
    _mm_store_si128(back_hash, _mm_add_epi32(CDGH, *back_hash));
}

std::vector<uint8_t> sha256_shani(std::vector<uint8_t>& data) {
    // 初期ハッシュ値
    static const uint32_t state[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };
    // sha256rnds2用にstateを並び変える
    static const __m128i A0 = _mm_shuffle_epi32(_mm_unpacklo_epi64(_mm_load_si128((__m128i const*)state + 0), _mm_load_si128((__m128i const*)state + 1)), 0x1B);
    static const __m128i C0 = _mm_shuffle_epi32(_mm_unpackhi_epi64(_mm_load_si128((__m128i const*)state + 0), _mm_load_si128((__m128i const*)state + 1)), 0x1B);
    // エンディアン変換に使用
    static const __m128i MASK = _mm_set_epi32(0x0c0d0e0f, 0x08090a0b, 0x04050607, 0x00010203);
    static const __m128i shuffleMask = _mm_set_epi8(12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3);

    // 計算に必要な変数の宣言
    uint8_t* data_ptr = data.data();
    int num_blocks = data.size() / 64;
    uint8_t hash[32]{};
    __m128i front_hash, back_hash, W0[4]{};
    int i;

    // 初期値のセット
    front_hash = A0;
    back_hash = C0;

    while (num_blocks > 0) {
        for (i = 0; i < 4; ++i) {
            W0[i] = _mm_shuffle_epi8(_mm_load_si128((__m128i const*)data_ptr + i), MASK);
        }
        oneblock_sha256(&front_hash, &back_hash, &W0[0]);

        data_ptr += 64;
        num_blocks--;
    }

    // 最終ブロックの処理
    int last_data_length = data.size() % 64;
    int last_data_Wnum = last_data_length / 16;
    uint8_t buf1[16]{}, buf2[16]{};

    for (i = 0; i < 4; ++i) {
        W0[i] = _mm_setzero_si128();
    }
    for (i = 0; i < last_data_Wnum; ++i) {
        W0[i] = _mm_shuffle_epi8(_mm_load_si128((__m128i const*)data_ptr + i), MASK);
    }

    int remainder_length = last_data_length % 16;
    if (remainder_length != 0) std::memcpy(buf1, (__m128i const*)data_ptr + i, remainder_length);
    buf1[remainder_length] = 0x80;

    if (last_data_length + 1 <= 48) {
        W0[i] = _mm_shuffle_epi8(_mm_load_si128((__m128i const*)buf1), MASK);
        for (i = 8; i > 0; --i) {
            buf2[16 - i] = static_cast<uint8_t>((data.size() * 8 >> ((i - 1) * 8)) & 0xFF);
        }
        W0[3] = _mm_shuffle_epi8(_mm_load_si128((__m128i const*)buf2), MASK);
        oneblock_sha256(&front_hash, &back_hash, &W0[0]);
    }
    else if (last_data_length + 1 <= 56) {
        for (i = 8; i > 0; --i) {
            buf1[16 - i] = static_cast<uint8_t>((data.size() * 8 >> ((i - 1) * 8)) & 0xFF);
        }
        W0[3] = _mm_shuffle_epi8(_mm_load_si128((__m128i const*)buf1), MASK);
        oneblock_sha256(&front_hash, &back_hash, &W0[0]);
    }
    else {
        W0[3] = _mm_shuffle_epi8(_mm_load_si128((__m128i const*)buf1), MASK);
        oneblock_sha256(&front_hash, &back_hash, &W0[0]);
        __m128i W_last[4]{};
        for (i = 8; i > 0; --i) {
            buf2[16 - i] = static_cast<uint8_t>((data.size() * 8 >> ((i - 1) * 8)) & 0xFF);
        }
        W_last[3] = _mm_shuffle_epi8(_mm_load_si128((__m128i const*)buf2), MASK);
        oneblock_sha256(&front_hash, &back_hash, &W_last[0]);
    }

    // 最終的なhashを代入
    __m128i X = _mm_shuffle_epi32(_mm_unpackhi_epi64(front_hash, back_hash), 0xB1);
    __m128i Y = _mm_shuffle_epi32(_mm_unpacklo_epi64(front_hash, back_hash), 0xB1);
    _mm_store_si128((__m128i*)hash + 0, _mm_shuffle_epi8(X, shuffleMask));
    _mm_store_si128((__m128i*)hash + 1, _mm_shuffle_epi8(Y, shuffleMask));

    return std::vector<uint8_t>(hash, hash + 32);
}